#
# Make file for LaneDetector
#

# Author: Mohamed Aly <malaa@caltech.edu>
# Date: 10/7/2010

OCVFLAGS = `pkg-config --cflags opencv`
OCVLIBS = `pkg-config --libs opencv`

CPP = g++-7

# type of system?
LBITS = $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)
	# do 64 bit stuff here, like set some CFLAGS
	SFX = 64
else
	SFX = 32
endif

# Add stop-line detection
SRCS    += CameraInfoOpt.c LaneDetectorOpt.c cmdline.c LaneDetector.cc \
	InversePerspectiveMapping.cc mcv.cc main.cc
OBJECTS += CameraInfoOpt.o LaneDetectorOpt.o cmdline.o \
	LaneDetector.o InversePerspectiveMapping.o mcv.o \
	main.o
CXXFLAGS += -std=c++17
INCLUDES += $(OCVFLAGS)
LIBS += $(OCVLIBS) -lstdc++
BINARY = LaneDetector$(SFX)

all: release

release: $(OBJECTS)
	$(CPP) $^ $(LDFLAGS) $(LIBS) $(CXXFLAGS) -O3 -o $(BINARY)

debug: $(OBJECTS)
	$(CPP) $^ $(LDFLAGS) $(LIBS) $(CXXFLAGS) -g -O0 -o $(BINARY)

# Generate getopts header
LaneDetectorOpt.c: LaneDetectorOpt.ggo
	gengetopt -i LaneDetectorOpt.ggo --conf-parser -F LaneDetectorOpt \
	  --func-name=LaneDetectorParser --arg-struct-name=LaneDetectorParserInfo

#get opts for cameraInfo and stopLinePerceptor
CameraInfoOpt.c: CameraInfoOpt.ggo
	gengetopt -i CameraInfoOpt.ggo -F CameraInfoOpt \
	  --func-name=cameraInfoParser \
	  --arg-struct-name=CameraInfoParserInfo \
	  --conf-parser

cmdline.c: cmdline.ggo
	gengetopt -i cmdline.ggo -u --conf-parser

clean:
	rm -f *.a $(OBJECTS) $(BINARY)
	rm -f CameraInfoOpt.c LaneDetectorOpt.c cmdline.c
	rm -f CameraInfoOpt.h LaneDetectorOpt.h cmdline.h

.cc.o:
	$(CPP) $< $(LDFLAGS) $(INCLUDES) $(CXXFLAGS) -c -o $@

# headers and sources
.hh.cc:
.h.c:

# generating gengetopt headers
cmdline.o: cmdline.c
CameraInfoOpt.o: CameraInfoOpt.c
LaneDetectorOpt.o: LaneDetectorOpt.c
