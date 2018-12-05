#ifndef ROS_UTILS_H_
#define ROS_UTILS_H_

#include "common_headers.h"
#include "lua_utils.h"

extern std::string pkg_dir;
extern std::string save_path;
extern int num_key_points;
extern double learning_rate;
extern double decay_rate;
extern double color_var;


void setRosParameters(const ros::NodeHandle& nh);
void msgCallback(const sensor_msgs::ImageConstPtr& img,
                 const darknet_ros_msgs::BoundingBoxesConstPtr& box,
                 const object_keypoint_msgs::ObjectKeyPointArrayConstPtr& kpa);



#endif
