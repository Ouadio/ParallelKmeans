# KMEANS ALGO
add_library(kmeansAlgo kmeans.cpp)

target_link_libraries(kmeansAlgo PRIVATE ${OpenCV_LIBS} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)

target_include_directories(kmeansAlgo PUBLIC ${PROJECT_SOURCE_DIR}/include)
