function merge = ccvCheckMergeSplines(spline1, spline2, meanDistThreshold, ...
  medianDistThreshold)
% CCVCHECKMERGESPLINES checks if two splines are to be merged (matched) or
% not. It does this by computing the distance from every point in each
% spline to every other point in the other spline and checking that the mean 
% or median distances are below the given thresholds.
%
% Inputs:
% -------
% spline2: Nx2 matrix of spline control points
% spline2: Nx2 matrix of spline control points
% meanDistThreshold: threshold for the mean distance between points on the
%   two splines
% medianDistThreshold: threshold for median distance between points on the
%   two splines
% 
% Outputs:
% --------
% merge: 1 if to merge, 0 otherwise
%

%get points on both
p1 = ccvEvalBezSpline(spline1, .01);
p2 = ccvEvalBezSpline(spline2, .01);

%now for every point in spline1, compute nearest in spline2, and get that
%distance
dist1 = zeros(1, size(p1,1));
for i=1:size(p1, 1)
    %get diff
    d = repmat(p1(i,:), size(p2, 1), 1) - p2;
    %get distance
    d = sqrt(sum(d.^2, 2));
    %get min
    dist1(i) = min(d);
end;
dist2 = zeros(1, size(p2,1));
for i=1:size(p2, 1)
    %get diff
    d = repmat(p2(i,:), size(p1, 1), 1) - p1;
    %get distance
    d = sqrt(sum(d.^2, 2));
    %get min
    dist2(i) = min(d);
end;

%compute mean and median
meanDist = min(mean(dist1), mean(dist2));
medianDist = min(median(dist1), median(dist2));

merge = (meanDist <= meanDistThreshold) || (medianDist <= medianDistThreshold);
