#include "ros_utils.h"

std::string pkg_dir;
std::string save_path;
int num_key_points;
double learning_rate;
double decay_rate;
double color_var;


void setRosParameters(const ros::NodeHandle& nh){

    nh.param("pkg_dir", pkg_dir, std::string("/home/rohan/rohan_m15x/ros_ws/src/train_hg"));
    nh.param("save_path", save_path, std::string("/home/rohan/tmp/train_weights"));
    nh.param("num_key_points", num_key_points, int(20));
    nh.param("learning_rate", learning_rate, double(0.00025));
    nh.param("decay_rate", decay_rate, double(0));
    nh.param("color_var", color_var, double(0.2));

    initializeLua();
    setLuaParameters();

    return;
}


void msgCallback(const sensor_msgs::ImageConstPtr& img,
                 const darknet_ros_msgs::BoundingBoxesConstPtr& box,
                 const object_keypoint_msgs::ObjectKeyPointArrayConstPtr& kpa){

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //read input image
    cv::Mat read_image;
    try{read_image = cv_bridge::toCvShare(img, "bgr8")->image;}
    catch (cv_bridge::Exception& e){ROS_ERROR("Could not convert from '%s' to 'bgr8'.", img->encoding.c_str());}

    //read input bounding box
    darknet_ros_msgs::BoundingBox bbox = box->bounding_boxes[0];

    //read input keypoint array
    unsigned int keypt_num = kpa->object[0].keypoint.size();
    double* keypt_array = new double[keypt_num*2*sizeof(double)];
    for(unsigned int i = 0; i < keypt_num*2; i++) {
        keypt_array[i] = kpa->object[0].keypoint[i/2].position.x;
        keypt_array[i+1] = kpa->object[0].keypoint[i/2].position.y;
        i++;
    }

    setInputImage(read_image);
    setInputBox(bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax);
    setInputKeyPoints(keypt_num, keypt_array);
    luaCallback();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "callBack time: " << duration/1000 << " ms" << std::endl;
    return;
}

