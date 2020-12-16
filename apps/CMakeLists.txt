#MAIN APP

set(Tbb_LIBS tbb)

# Main executables setting & linking
set(apps app benchmark)

foreach( TAR ${apps} )
    add_executable(${TAR} "${TAR}.cpp")
    target_include_directories(${TAR} PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${TAR} PUBLIC kmeansAlgo ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Tbb_LIBS})
endforeach(TAR)

