package core.game;

import core.competition.CompetitionParameters;
import core.player.Player;
import core.vgdl.SpriteGroup;
import core.vgdl.VGDLFactory;
import core.vgdl.VGDLParser;
import core.vgdl.VGDLRegistry;
import ontology.Types;
import tools.ElapsedCpuTimer;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;

/**
 * JPype bridge for GVGAI game engine.
 *
 * Exposes a step-by-step game API without socket/subprocess overhead,
 * allowing Python (via JPype) to drive the game loop directly.
 *
 * Usage from Python:
 *   bridge = GVGAIBridge(gameFile, levelFiles)
 *   bridge.reset(levelIdx=0, randomSeed=42)
 *   bridge.step("ACTION_RIGHT")
 *   score = bridge.getGameScore()
 *   done  = bridge.isGameOver()
 */
public class GVGAIBridge {

    // ------------------------------------------------------------------ //
    // Fields
    // ------------------------------------------------------------------ //

    private Game game;
    private final String gameFile;
    private final String[] levelFiles;
    private double lastScore;
    private PythonPlayer pythonPlayer;
    private Dimension screenSize;
    private int currentSeed = 0;

    /** Cached observation JSON string (populated by step / reset). */
    private String lastObsJSON = "";

    // ------------------------------------------------------------------ //
    // Inner class: Python-controlled player
    // ------------------------------------------------------------------ //

    /**
     * Minimal Player whose action is set externally (from Python)
     * before each game cycle.
     */
    static class PythonPlayer extends Player {
        private Types.ACTIONS pendingAction = Types.ACTIONS.ACTION_NIL;

        public void setAction(Types.ACTIONS action) {
            this.pendingAction = action;
        }

        @Override
        public Types.ACTIONS act(StateObservation obs, ElapsedCpuTimer timer) {
            return pendingAction;
        }

        @Override
        public Types.ACTIONS act(StateObservationMulti obs, ElapsedCpuTimer timer) {
            return pendingAction;
        }
    }

    // ------------------------------------------------------------------ //
    // Constructor
    // ------------------------------------------------------------------ //

    /**
     * Creates a bridge for a given game.
     *
     * VGDLFactory / VGDLRegistry are initialised once (idempotent).
     *
     * @param gameFile   absolute path to the VGDL game description (.txt)
     * @param levelFiles array of absolute paths to level files (index = level id)
     */
    public GVGAIBridge(String gameFile, String[] levelFiles) {
        VGDLFactory.GetInstance().init();
        VGDLRegistry.GetInstance().init();
        // This bridge is used by the Gym/JPype inference environment, not by the
        // Java-side learning pipeline. Enabling learning mode changes engine
        // behaviour and can break normal level loading for some games.
        CompetitionParameters.IS_LEARNING = false;

        this.gameFile    = gameFile;
        this.levelFiles  = levelFiles;
        this.pythonPlayer = new PythonPlayer();
    }

    // ------------------------------------------------------------------ //
    // Public API
    // ------------------------------------------------------------------ //

    /**
     * Resets the game to the given level.
     *
     * @param levelIdx   0-based index into levelFiles
     * @param randomSeed seed for the game's RNG
     */
    public void reset(int levelIdx, int randomSeed) {
        this.currentSeed = randomSeed;
        // Parse fresh game from VGDL
        this.game = new VGDLParser().parseGame(gameFile);
        this.game.buildLevel(levelFiles[levelIdx], randomSeed);
        this.game.initRandomGenerator(randomSeed);
        this.lastScore = 0.0;
        this.screenSize = game.getScreenSize();

        // ---- Replicate Game.prepareGame() (private) ----
        game.gameTick = -1;
        game.isEnded  = false;
        game.createAvatars(0);   // humanID = 0

        // Assign our pythonPlayer to every avatar
        if (game.avatars != null) {
            for (int i = 0; i < game.avatars.length; i++) {
                if (game.avatars[i] != null) {
                    game.avatars[i].player = pythonPlayer;
                    game.avatars[i].setPlayerID(i);
                }
            }
        }
        game.initForwardModel();
        // Defensive: some engine paths can clear RNG during init.
        if (game.getRandomGenerator() == null) {
            game.initRandomGenerator(randomSeed);
        }
        // -----------------------------------------------

        lastObsJSON = buildObsJSON();
    }

    /**
     * Advances the game by one step using the supplied action.
     *
     * @param actionName one of ACTION_NIL / ACTION_UP / ACTION_DOWN /
     *                   ACTION_LEFT / ACTION_RIGHT / ACTION_USE
     */
    public void step(String actionName) {
        if (game == null || game.isEnded) return;

        // Defensive: ensure RNG is always available before ticking.
        if (game.getRandomGenerator() == null) {
            game.initRandomGenerator(currentSeed);
        }

        Types.ACTIONS action = Types.ACTIONS.fromString(actionName);
        pythonPlayer.setAction(action);

        // ---- Replicate Game.gameCycle() (private) ----
        game.gameTick++;
        game.fwdModel.update(game);
        game.tick();
        game.eventHandling();
        game.clearAll(game.fwdModel);
        game.terminationHandling();
        game.checkTimeOut();
        // ----------------------------------------------

        lastObsJSON = buildObsJSON();
    }

    // ------------------------------------------------------------------ //
    // State accessors
    // ------------------------------------------------------------------ //

    public boolean isGameOver() {
        return game == null || game.isEnded;
    }

    public double getGameScore() {
        if (game == null || game.avatars == null || game.avatars[0] == null)
            return 0.0;
        return game.avatars[0].getScore();
    }

    /** Score delta since the previous step (reward signal). */
    public double getScoreDelta() {
        double current = getGameScore();
        double delta   = current - lastScore;
        lastScore      = current;
        return delta;
    }

    public String getWinner() {
        if (game == null || game.avatars == null || game.avatars[0] == null)
            return "NO_WINNER";
        return game.avatars[0].getWinState().toString();
    }

    public int getGameTick() {
        return game != null ? game.gameTick : -1;
    }

    /**
     * Returns the list of actions available for the avatar,
     * excluding ACTION_NIL (index 0 in the Gym action space).
     */
    public String[] getAvailableActions() {
        if (game == null) return new String[0];
        StateObservation obs = game.getObservation();
        ArrayList<Types.ACTIONS> actions = obs.getAvailableActions();
        String[] result = new String[actions.size()];
        for (int i = 0; i < actions.size(); i++) {
            result[i] = actions.get(i).toString();
        }
        return result;
    }

    /**
     * Returns a compact ASCII representation of the current observation grid.
     *
     * Format mirrors the socket-based observationString:
     *   rows separated by '\n', columns separated by ','
     *   each cell contains space-separated itypeKey tokens.
     */
    public String getObservationString() {
        if (game == null) return "";
        StateObservation obs = game.getObservation();
        ArrayList<Observation>[][] grid = obs.getObservationGrid();
        if (grid == null) return "";

        int cols = grid.length;
        if (cols == 0) return "";
        int rows = grid[0].length;

        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < rows; r++) {
            if (r > 0) sb.append('\n');
            for (int c = 0; c < cols; c++) {
                if (c > 0) sb.append(',');
                ArrayList<Observation> cell = grid[c][r];
                if (cell != null) {
                    for (int s = 0; s < cell.size(); s++) {
                        if (s > 0) sb.append(' ');
                        sb.append(cell.get(s).itypeKey);
                    }
                }
            }
        }
        return sb.toString();
    }

    /**
     * Returns the avatar's pixel position as [x, y].
     * Returns [-1, -1] if not available.
     */
    public double[] getAvatarPosition() {
        if (game == null) return new double[]{-1, -1};
        StateObservation obs = game.getObservation();
        tools.Vector2d pos = obs.getAvatarPosition();
        return new double[]{pos.x, pos.y};
    }

    /**
     * Returns the block size (pixels per grid cell).
     */
    public int getBlockSize() {
        if (game == null) return 0;
        return game.getObservation().getBlockSize();
    }

    /**
     * Returns a full JSON-serialised state observation (mirrors the socket
     * protocol's JSON payload).
     */
    public String getObservationJSON() {
        return lastObsJSON;
    }

    // ------------------------------------------------------------------ //
    // Rendering
    // ------------------------------------------------------------------ //

    /**
     * Renders the current game frame to a PNG byte array.
     * Uses Java2D, so works in headless environments (no display required).
     *
     * @return PNG-encoded bytes, or empty array on failure.
     */
    public byte[] renderToBytes() {
        if (game == null || screenSize == null) return new byte[0];
        try {
            int w = (int) screenSize.getWidth();
            int h = (int) screenSize.getHeight();
            BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = bi.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Black background
            g.setColor(Color.BLACK);
            g.fillRect(0, 0, w, h);

            // Draw all sprites following sprite order
            int[] spriteOrder = game.getSpriteOrder();
            for (int spriteTypeInt : spriteOrder) {
                SpriteGroup sg = game.spriteGroups[spriteTypeInt];
                if (sg != null) {
                    ArrayList<core.vgdl.VGDLSprite> sprites = sg.getSprites();
                    if (sprites != null) {
                        for (core.vgdl.VGDLSprite sp : sprites) {
                            if (sp != null) sp.draw(g, game);
                        }
                    }
                }
            }
            g.dispose();

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(bi, "png", baos);
            return baos.toByteArray();
        } catch (Exception e) {
            e.printStackTrace();
            return new byte[0];
        }
    }

    // ------------------------------------------------------------------ //
    // Internal helpers
    // ------------------------------------------------------------------ //

    private String buildObsJSON() {
        if (game == null) return "{}";
        try {
            StateObservation obs = game.getObservation();
            SerializableStateObservation sso = new SerializableStateObservation(obs);
            tools.com.google.gson.Gson gson = new tools.com.google.gson.Gson();
            return gson.toJson(sso);
        } catch (Exception e) {
            return "{}";
        }
    }
}
