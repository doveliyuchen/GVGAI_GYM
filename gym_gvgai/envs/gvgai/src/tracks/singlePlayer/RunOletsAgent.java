package tracks.singlePlayer;

import tracks.ArcadeMachine;

public class RunOletsAgent {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java tracks.singlePlayer.RunOletsAgent <game_file> <level_file> [visuals_on]");
            System.exit(1);
        }

        String gameFile = args[0];
        String levelFile = args[1];
        boolean visuals = false;
        if (args.length > 2 && args[2].equalsIgnoreCase("true")) {
            visuals = true;
        }

        String oletsAgent = "tracks.singlePlayer.advanced.olets.Agent";
        int seed = (int) (Math.random() * 1000);

        // Run the game
        ArcadeMachine.runOneGame(gameFile, levelFile, visuals, oletsAgent, null, seed, 0);
    }
}
