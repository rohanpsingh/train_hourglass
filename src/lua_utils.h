#ifndef LUA_UTILS_H_
#define LUA_UTILS_H_

#include "common_headers.h"
#include "ros_utils.h"

extern "C" {
    #include <lua.h>
    #include <lualib.h>
    #include <lauxlib.h>
    #include <luaT.h>
    #include <TH/TH.h>
}

//lua state
extern lua_State *L;

void initializeLua();
void setLuaParameters();
void setInputImage(const cv::Mat& input_image);
void setInputBox(const int& xmin, const int& xmax, const int& ymin, const int& ymax);
void setInputKeyPoints(const int& num_kpts, double keypt_data[]);
void loadLuaInputData();
void trainOnBatch();
#endif
