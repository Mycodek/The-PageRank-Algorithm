main:
	mpic++ -m64 -g -O -I../mrmpi/mrmpi-7Apr14/src  -c mr-pr-mpi-base.cpp
	mpic++ -g -O mr-pr-mpi-base.o ../mrmpi/mrmpi-7Apr14/src/libmrmpi_mpicc.a  -o mr-pr-mpi-base
	g++ mr-pr-cpp.cpp /usr/lib/x86_64-linux-gnu/libboost_system.a /usr/lib/x86_64-linux-gnu/libboost_iostreams.a /usr/lib/x86_64-linux-gnu/libboost_filesystem.a -pthread
	mpic++ mr-pr-mpi.cpp
