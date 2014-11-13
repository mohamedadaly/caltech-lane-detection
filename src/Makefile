#
# Make file for LaneDetector
#

# Author: Mohamed Aly <malaa@caltech.edu>
# Date: 10/7/2010

OCVFLAGS = `pkg-config --cflags opencv`
OCVLIBS = `pkg-config --libs opencv` -lstdc++

CPP = g++

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
CFLAGS += $(OCVFLAGS)
LIBS += $(OCVLIBS)
BINARY = LaneDetector$(SFX)

all: release

release: $(OBJECTS)
	$(CPP) $^ $(LDFLAGS) $(LIBS) $(CFLAGS) -O3 -o $(BINARY)

debug: $(OBJECTS)
	$(CPP) $^ $(LDFLAGS) $(LIBS) $(CFLAGS) -g -O0 -o $(BINARY)

# Generate getopts header
LaneDetectorOpt.h: LaneDetectorOpt.ggo
	gengetopt -i LaneDetectorOpt.ggo --conf-parser -F LaneDetectorOpt \
	  --func-name=LaneDetectorParser --arg-struct-name=LaneDetectorParserInfo

#get opts for cameraInfo and stopLinePerceptor
cameraInfoOpt.h: cameraInfoOpt.ggo
	gengetopt -i cameraInfoOpt.ggo -F cameraInfoOpt \
	  --func-name=cameraInfoParser \
	  --arg-struct-name=CameraInfoParserInfo \
	  --conf-parser

cmdline.h: cmdline.ggo
	gengetopt -i cmdline.ggo -u --conf-parser

clean:
	rm -f *.a $(OBJECTS) $(BINARY)

.cc.o:
	g++ $< $(CFLAGS) $(LIBS) $(LDFLAGS) -c -o $@

# headers and sources
.hh.cc:
.h.c:

# generating gengetopt headers
cmdline.o: cmdline.h
CameraInfoOpt.o: CameraInfoOpt.h
LaneDetectorOpt.o: LaneDetectorOpt.h
