
# Python
# Expert Recrder App
import argparse
import getch
import random
import gym
import numpy as np
import time
import os

#GLOBALS
BINDINGS={
    'a':0,      #left
    'b':2,      #right
    'c':1,      #nothing

}
SHARD_SIZE=2000

def get_options():
    parser = argparse.ArgumentParser(description='Records an expert...')
    parser.add_argument('data_directory',type=str,
    help="The main datastore fo rthis perticular expert.")
    args=parser.parse_args()
    print("::print::args=",args)
    return args
def run_recorder(opts):
    # print(opts)
    """
    Runs the main recorder by binding certain deiscrete actions to keys.
    """
    ddir=opts.data_directory
    record_history = []             #state action hist buffer
    env=gym.make('MountainCar-v0')
    env._max_episode_steps=1200    #how many chances to make to top

    #############
    # Bind Keys #
    #############

    action=None
    esc=False
    shard_suffix=''.join(random.choice('0123456789ABCEF') for i in range(16))
    sarsa_pairs=[]

    while not esc:
        done=False
        _last_obs=env.reset()

        while not done:
            env.render()
            # Handle toggling of diff application states

            # Take current action if key pressed
            action=None
            while action is None:
                keys_pressed=getch.getch()
                if keys_pressed is '+':
                    esc=True
                    break
                # cont..
                pressed=[x for x in BINDINGS if x in keys_pressed]
                # (below)if valid input detectd then store in action else None
                action=BINDINGS[ pressed[0] ] if len(pressed)>0 else None

            if esc:
                print("Ending (esc/+ prssed)")
                done=True #ensure outer loop is bypassed
                break

            obs,reward,done,info=env.step(action)
            no_action=False
            sarsa=(_last_obs,action) # (current_stat,action_to_take)
            _last_obs=obs
            sarsa_pairs.append(sarsa)

        if esc:
            break

    print("SAVING")
    # Save out recording data
    num_shards=int(np.ceil(len(sarsa_pairs)/SHARD_SIZE))
    for shard_iter in range(num_shards):
        shard = sarsa_pairs[
            shard_iter*SHARD_SIZE: min(
                (shard_iter+1)*SHARD_SIZE, len(sarsa_pairs)
            )
        ]
        shard_name="{}_{}.npy".format(str(shard_iter), shard_suffix)
        with open(os.path.join(ddir, shard_name), 'wb')as f:
            np.save(f, sarsa_pairs)

if __name__ == "__main__":
    # get_options()
    run_recorder(get_options())

