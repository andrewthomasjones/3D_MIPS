# CMakeLists.txt
 
cmake_minimum_required(VERSION 2.8)
 
project(MIPSConv)
find_package(ITK REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Boost 
	COMPONENTS 	filesystem system
 	REQUIRED)

include_directories(${BOOST_INCLUDEDIR})
link_directories(${BOOST_LIBRARYDIR}) 

include(${ITK_USE_FILE})

SET(GCC_COMPILE_FLAGS "-std=gnu++11 -fpermissive")
add_definitions(${GCC_COMPILE_FLAGS})

add_executable(MIPSConv MIPSConv.cxx)

target_link_libraries(MIPSConv
	${Glue} 
	${ITK_LIBRARIES} 
	${OpenCV_LIBS} 
	${Boost_LIBRARIES}
)
 


