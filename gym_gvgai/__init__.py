from gymnasium.envs.registration import register, make
import os

dir = os.path.dirname(__file__)
gamesPath = os.path.join(dir, os.path.normpath('envs/games'))
games = os.listdir(gamesPath)

for game in games:
    gamePath = os.path.join(gamesPath, game)
    if os.path.isdir(gamePath):
        lvls = len([lvl for lvl in os.listdir(gamePath) if 'lvl' in lvl])
        for lvl in range(lvls):
            name    = game.split('_')[0]
            version = int(game.split('_')[-1][1:])

            # ----------------------------------------------------------------
            # JPype-backed environment (primary, no subprocess or socket)
            # ----------------------------------------------------------------
            register(
                id=f'gvgai-{name}-lvl{lvl}-v{version}',
                entry_point='gym_gvgai.envs.gvgai_env_jpype:GVGAI_Env_JPype',
                kwargs={
                    'game':     name,
                    'level':    lvl,
                    'version':  version,
                    'base_dir': os.path.join(dir, 'envs'),
                },
                max_episode_steps=2000,
            )

            # NOTE: legacy socket env registration removed.
            # This package is Gymnasium-only and JPype-only now.
