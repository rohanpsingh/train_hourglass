project(train_hg)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14" )

set(LUA_INCLUDE_DIR ~/torch/install/include/)
set(LUA_LIBRARY ~/torch/install/lib/libluajit.so ~/torch/install/lib/libluaT.so ~/torch/install/lib/libTH.so)

find_package(PCL 1.7 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    object_keypoint_msgs
    cv_bridge
    image_transport
    darknet_ros_msgs
    )
find_package(OpenCV REQUIRED)

include_directories(include
    ${PCL_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${LUA_INCLUDE_DIR}
    )

catkin_package(
   CATKIN_DEPENDS
        roscpp
        std_msgs
        sensor_msgs
        geometry_msgs
	object_keypoint_msgs	
)


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(train src/main_node.cpp src/ros_utils.cpp src/lua_utils.cpp)
target_link_libraries(train ${catkin_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} opencv_core opencv_highgui ${LUA_LIBRARY})
