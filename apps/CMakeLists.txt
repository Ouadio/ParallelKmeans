#MAIN APP

add_executable(app app.cpp)

target_link_libraries(app PUBLIC kmeansAlgo ${OpenCV_LIBS} ${Boost_LIBRARIES})

target_include_directories(app PUBLIC ${PROJECT_SOURCE_DIR}/include)
