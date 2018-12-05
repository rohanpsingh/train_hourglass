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

    cv::Mat read_image, image;
    try{
        read_image = cv_bridge::toCvShare(img, "bgr8")->image;
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", img->encoding.c_str());
    }
    read_image.convertTo(image, CV_32FC3);

    cv::Mat img_bgr[3];
    cv::split(image, img_bgr);

    const int height = read_image.rows;
    const int width = read_image.cols;
    const int channs = read_image.channels();
    const int size = channs*height*width;
    lua_getglobal(L, "inImage_c1");
    float* imgData = img_bgr[0].ptr<float>();
    THFloatStorage* imgStorage = THFloatStorage_newWithData(imgData, size);
    THFloatTensor* imgTensor = THFloatTensor_newWithStorage2d(imgStorage, 0, height, width, width, 1);
    luaT_pushudata(L, (void*)imgTensor, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c1");

    lua_getglobal(L, "inImage_c2");
    float* imgData1 = img_bgr[1].ptr<float>();
    THFloatStorage* imgStorage1 = THFloatStorage_newWithData(imgData1, size);
    THFloatTensor* imgTensor1 = THFloatTensor_newWithStorage2d(imgStorage1, 0, height, width, width, 1);
    luaT_pushudata(L, (void*)imgTensor1, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c2");

    lua_getglobal(L, "inImage_c3");
    float* imgData2 = img_bgr[2].ptr<float>();
    THFloatStorage* imgStorage2 = THFloatStorage_newWithData(imgData2, size);
    THFloatTensor* imgTensor2 = THFloatTensor_newWithStorage2d(imgStorage2, 0, height, width, width, 1);
    luaT_pushudata(L, (void*)imgTensor2, "torch.FloatTensor");
    lua_setglobal(L, "inImage_c3");

    const int xmin = box->bounding_boxes[0].xmin;
    const int xmax = box->bounding_boxes[0].xmax;
    const int ymin = box->bounding_boxes[0].ymin;
    const int ymax = box->bounding_boxes[0].ymax;
    int cx = abs(xmax+xmin)/2;
    int cy = abs(ymax+ymin)/2;
    float scale = std::max(xmax-xmin, ymax-ymin);
    scale /= 200.0f;


    lua_getglobal(L,"input_scale");
    lua_pushnumber(L,scale);
    lua_setglobal(L,"input_scale");

    lua_getglobal(L,"input_center_x");
    lua_pushnumber(L,cx);
    lua_setglobal(L,"input_center_x");

    lua_getglobal(L,"input_center_y");
    lua_pushnumber(L,cy);
    lua_setglobal(L,"input_center_y");


    lua_getglobal(L,"input_parts");
    unsigned int num_kpts = kpa->object[0].keypoint.size();
    double* keypt_data = new double[num_kpts*2*sizeof(double)];
    for(unsigned int i = 0; i < num_kpts*2; i++) {
        keypt_data[i] = kpa->object[0].keypoint[i/2].position.x;
        keypt_data[i+1] = kpa->object[0].keypoint[i/2].position.y;
        i++;
    }
    THDoubleStorage* kptStorage = THDoubleStorage_newWithData(keypt_data, num_kpts*2);
    THDoubleTensor* kptTensor = THDoubleTensor_newWithStorage2d(kptStorage, 0, num_kpts, 2, 2, 1);
    luaT_pushudata(L, (void*)kptTensor, "torch.DoubleTensor");
    lua_setglobal(L,"input_parts");


    lua_pcall(L,3,0,1);
    lua_getglobal(L, "init");
    lua_pcall(L,0,0,0);

    lua_getglobal(L, "trainOnThis");
    lua_pcall(L,0,0,0);

    lua_gc(L, LUA_GCCOLLECT, 0);
    lua_settop(L, 0);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "callBack time: " << duration/1000 << " ms" << std::endl;
    return;
}

