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

