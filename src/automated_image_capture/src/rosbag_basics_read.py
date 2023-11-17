import rosbag
bag = rosbag.Bag('test.bag')
for topic, msg, t in bag.read_messages(topics=['/camera/image']):
    #print(msg)
bag.close()
