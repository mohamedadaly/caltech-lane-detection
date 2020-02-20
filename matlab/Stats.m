function Stats
% STATS computes stats for the results of LaneDetector
%
% Inputs:
% -------
% 
% Outputs:
% --------
%

% The detection files
detectionFiles = {
  '../clips/cordova1/list.txt_results.txt'
  '../clips/cordova2/list.txt_results.txt'
  '../clips/washington1/list.txt_results.txt'
  '../clips/washington2/list.txt_results.txt'
  };

% The ground truth labels
truthFiles = {
  '../clips/cordova1/labels.ccvl'
  '../clips/cordova2/labels.ccvl'
  '../clips/washington1/labels.ccvl'
  '../clips/washington2/labels.ccvl'
  };

% Get statistics
ccvGetLaneDetectionStats(detectionFiles, truthFiles);
