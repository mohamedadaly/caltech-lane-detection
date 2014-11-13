function [frames] = ccvReadLaneDetectionResultsFile(filename)
% CCVREADLANEDETECTIONRESULTSFILE reads a results file from the binary
% LaneDetector saved with --save-lanes flag
%
% INPUTS
% ------
% filename - the input results file
%
% OUTPUTS
% -------
% frames  - the output structure array, one per frame (image) with fields
%   .id - the frame id
%   .splines - the splines cell array
%   .numSplines - the number of splines
%   .scores - the scores of the splines
%
% See also 
%

%open file
file = fopen(filename, 'r');

%loop on file and read data
frames = [];
while 1
    %read frame
    d = fscanf(file, 'frame#%08d has %d splines\n', 2);
    %if no frames, then exit
    if isempty(d), break; end;
    
    %get id and number of splines
    id = d(1);
    numSplines = d(2);
    
    %now loop for this amount
    frame = [];
    splines = {};
    scores = [];
    for i=1:numSplines
        %get header
        d = fscanf(file, '\tspline#%d has %d points and score %f\n');
        if isempty(d), continue; end;
        scores = [scores, d(3)];
     
        %get spline points
        d = fscanf(file, '\t\t%f, %f\n');
        if isempty(d), continue, end;        
        spline = reshape(d, 2, [])';
        
        %put spline
        splines = [splines, spline];        
    end;
    
    %put frame
    frame.id = id;
    frame.splines = splines;
    frame.numSplines = length(splines);
    frame.scores = scores;
    frames = [frames, frame];
    
end;
