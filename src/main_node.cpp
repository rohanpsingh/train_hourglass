#include "common_headers.h"
#include "ros_utils.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


int main (int argc, char** argv){

    ros::init(argc, argv, "train");
    ros::NodeHandle nh("~");

    setRosParameters(nh);

    message_filters::Subscriber<sensor_msgs::Image> img_sub(nh, "input_image", 1);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> box_sub(nh, "input_bbox", 1);
    message_filters::Subscriber<object_keypoint_msgs::ObjectKeyPointArray> kpa_sub(nh, "input_kpts", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                            darknet_ros_msgs::BoundingBoxes,
                                                            object_keypoint_msgs::ObjectKeyPointArray> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(100), img_sub, box_sub, kpa_sub);
    sync.registerCallback(boost::bind(&msgCallback, _1, _2, _3));


    ros::spin();

    return 0;
}
