# KMEANS ALGO
add_library(kmeansAlgo kmeans.cpp)
add_library(kmeansMPIAlgo kmeansMPI.cpp)

target_link_libraries(kmeansAlgo PRIVATE ${OpenCV_LIBS} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(kmeansMPIAlgo PRIVATE ${OpenCV_LIBS} ${Boost_LIBRARIES})

target_include_directories(kmeansAlgo PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(kmeansMPIAlgo PUBLIC ${PROJECT_SOURCE_DIR}/include)
