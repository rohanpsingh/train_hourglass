#include "lua_utils.h"

lua_State *L;


void initializeLua(){
    L = luaL_newstate();
    std::cout << "------lua loading libraries----- " << std::endl;
    luaL_openlibs(L);
    int status;
    status = luaL_loadfile(L, std::string(pkg_dir + "/src/lua/init.lua").c_str());
    if (lua_pcall(L, 0, LUA_MULTRET, 0)) {
        fprintf(stderr, "Failed to run script: %s\n", lua_tostring(L, -1));
        exit(1);
    }
    status = luaL_loadfile(L, std::string(pkg_dir + "/src/lua/main.lua").c_str());
    if (lua_pcall(L, 0, LUA_MULTRET, 0)) {
        fprintf(stderr, "Failed to run script: %s\n", lua_tostring(L, -1));
        exit(1);
    }

    return;
}


void setLuaParameters(){
    std::cout << "------setting parameters----- " << std::endl;
    lua_getglobal(L, "savePath_");
    lua_pushstring(L, save_path.c_str());
    lua_setglobal(L, "savePath_");
    lua_getglobal(L, "optNumParts_");
    lua_pushnumber(L, num_key_points);
    lua_setglobal(L, "optNumParts_");
    lua_getglobal(L, "optLR_");
    lua_pushnumber(L, learning_rate);
    lua_setglobal(L, "optLR_");
    lua_getglobal(L, "optLRdecay_");
    lua_pushnumber(L, decay_rate);
    lua_setglobal(L, "optLRdecay_");
    lua_getglobal(L, "optColorVar_");
    lua_pushnumber(L, color_var);
    lua_setglobal(L, "optColorVar_");

    return;
}


void setInputImage(const cv::Mat& input_image){
    cv::Mat image;
    input_image.convertTo(image, CV_32FC3);
    cv::Mat img_bgr[3];
    cv::split(image, img_bgr);

    const int height = input_image.rows;
    const int width = input_image.cols;
    const int channs = input_image.channels();
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

    return;
}


void setInputBox(const int& xmin, const int& xmax, const int& ymin, const int& ymax){
    const int cx = abs(xmax+xmin)/2;
    const int cy = abs(ymax+ymin)/2;
    const float scale = std::max(xmax-xmin, ymax-ymin)/200.0f;

    lua_getglobal(L,"input_scale");
    lua_pushnumber(L,scale);
    lua_setglobal(L,"input_scale");

    lua_getglobal(L,"input_center_x");
    lua_pushnumber(L,cx);
    lua_setglobal(L,"input_center_x");

    lua_getglobal(L,"input_center_y");
    lua_pushnumber(L,cy);
    lua_setglobal(L,"input_center_y");

    return;
}


void setInputKeyPoints(const int& num_kpts, double* keypt_data){
    lua_getglobal(L,"input_keypt");
    THDoubleStorage* kptStorage = THDoubleStorage_newWithData(keypt_data, num_kpts*2);
    THDoubleTensor* kptTensor = THDoubleTensor_newWithStorage2d(kptStorage, 0, num_kpts, 2, 2, 1);
    luaT_pushudata(L, (void*)kptTensor, "torch.DoubleTensor");
    lua_setglobal(L,"input_keypt");

    return;
}


void luaCallback(){

    lua_getglobal(L, "loadInputData");
    lua_pcall(L,0,0,0);

    lua_getglobal(L, "trainOnThis");
    lua_pcall(L,0,0,0);

    lua_gc(L, LUA_GCCOLLECT, 0);
    lua_settop(L, 0);

    return;
}
