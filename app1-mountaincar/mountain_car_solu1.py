# Python
import tensorflow as tf
import numpy as np
import argparse
import os
import gym

# Glboals
BINDINGS ={
    'w':1,
    'a':3,
    's':4,
    'd':2
}
SHARD_SIZE=200

def get_options():
    parser=argparse.ArgumentParser(description='Clone some expert data...')
    parser.add_argument('bc_data',type=str,
    help="The main datastore for this particular expert.")

    args=parser.parse_args()
    # print(args)
    return args

def process_data(bc_data_dir):
    """
    Runs training for the agent.
    """
    # Load
    # In future move to seperate thread?
    states, actions = [], []
    shards= [ x for x in os.listdir(bc_data_dir) if x.endswith('.npy') ]
    print("Processing shards: {}".format(shards))
    for shard in shards:
        shard_path = os.path.join(bc_data_dir,shard)
        with open(shard_path, 'rb') as f:
            data = np.load(f)
            shard_states, unprocessed_actions=zip(*data)
            shard_states=[ x.flatten() for x in shard_states ]

            # Add the shard to the dataset
            # There is an issue with append
            # but the workaroud is to use 1 big file!
            states.extend(shard_states)
            actions.extend(unprocessed_actions)
            # states.append(shard_states)
            # actions.append(unprocessed_actions)

        states=np.asarray(states, dtype=np.float32)
        actions=np.asarray(actions,dtype=np.float32)/2
        print("Processed with {} paris".format(len(states)))
        # print("states: ",states)
        # print ("actions: ",actions)
        return states,actions

def create_model():
    """
    Creates the model.
    """
    state_ph=tf.placeholder(tf.float32,shape=[None,2])
    # Process

    # Hidden
    with tf.variable_scope("layer1"):
        hidden=tf.layers.dense(state_ph,128,activation=tf.nn.relu)

    with tf.variable_scope("layer2"):
        hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu)
    # Make output layers
    with tf.variable_scope("layer3"):
        logits=tf.layers.dense(hidden,2)
    # Take the action with the highest activation
    with tf.variable_scope("output"):
        action=tf.argmax(input=logits,axis=1)

    return state_ph,action,logits
def create_training(logits):
    """
    Create the model
    """
    label_ph=tf.placeholder(tf.int32,shape=[None])
    #Convert it to a onehot 1-> [1,0,0,0]
    with tf.variable_scope("loss"):
        onehot_labels=tf.one_hot(indices=tf.cast(label_ph,tf.int32),depth=2)#4actions

        loss=tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        loss=tf.reduce_mean(loss)

        tf.summary.scalar('loss',loss)

        with tf.variable_scope("training"):
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-3)
            train_op=optimizer.minimize(loss=loss)

        return train_op, loss, label_ph

def run_main(opts):
    # Create the environment with specified argument
    state_data, action_data = process_data(opts.bc_data)
    env=gym.make('MountainCar-v0')
    env._max_episode_steps=1200

    x,model, logits=create_model()
    train,loss,labels=create_training(logits)

    sess = tf.Session()
    # Create summaries
    merged = tf.summary.merge_all()
    train_writer=tf.summary.FileWriter('./logs/logs',sess.graph) #modified

    sess.run(tf.global_variables_initializer())

    tick=0

    episode_cnt=0
    timestamp_cnt=0
    reward_curr_cnt=0

    while True:
        done=False
        obs=env.reset()

        episode_cnt= episode_cnt + 1
        timestamp_cnt=0
        reward_curr_cnt=0

        while not done:
            env.render()
            # Get random batch from data
            batch_index=np.random.choice(len(state_data),64)#batch size
            state_batch,action_batch=state_data[batch_index],action_data[batch_index]

            # Train model
            _, cur_loss, cur_summaries=sess.run([train,loss,merged],feed_dict={
                x: state_data,
                labels: action_data
            })
            print("Loss: {}".format(cur_loss))
            train_writer.add_summary(cur_summaries,tick)

            #Handle the toggling of diff app states
            action=sess.run(model, feed_dict={
                x: [obs.flatten()]
            }) [0]*2 # double check this part??

            obs,reward,done,info=env.step(action)
            tick += 1

            reward_curr_cnt= reward_curr_cnt +reward
            timestamp_cnt= timestamp_cnt+1
            if done:
                print("Episode Count             = ",episode_cnt)
                print("Current Reward/timestpes (Goal:MIN) = ",reward_curr_cnt)
                # print("Timestamp count           = ",timestamp_cnt)


def tester():
    print("tester!")

if __name__ == "__main__":
    # tester()
    inputDirectory=get_options()
    # tester
    # process_data(inputDirectory.bc_data)
    run_main(inputDirectory)
