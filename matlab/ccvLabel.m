function varargout = ccvLabel(f, varargin)
% CCVLABEL performs different tasks on the label structure, like creating
% new structure, adding frames, labels, ...etc.
% 
% INPUTS
% ------
% f         - the input function to perform
% varargin  - the rest of the inputs (potentially zero)
% 
% OUTPUTS
% -------
% varargout - the outputs from the selected operation
%
% See also ccvLabeler
% 
% AUTHOR    - Mohamed Aly <malaa@caltech.edu>
% DATE      - May 26, 2009
%

%check if we have a valid input function
if isempty(f) || ~exist(f, 'file'), error('Please enter a valid function'); end;

%call the function
varargout = cell(1, nargout);
[varargout{:}] = feval(f, varargin{:});
end

function ld = create()
% NEW creates a new empty structure
%
% INPUTS
% ------
% 
% OUTPUTS
% -------
% ld      -  the output empty label data
%

ld.version = 0;
ld.source = 'image';
ld.frames = struct('frame', {}, ...
   'labels', struct('points',{}, 'type',{}, 'subtype',{}, 'obj',{}));
ld.objects = struct('id',{});
end

function ld = read(fname) 
% READ loads label data from a file
%
% INPUTS
% ------
% fname   - the input file name
% 
% OUTPUTS
% -------
% ld      - the output empty label data
%

%load the file
ld = [];
try
  load(fname, '-mat');
catch
  return;
end;

%check version
if ~exist('ld', 'var') || ~ld.version<0
  error('invalid input file');
end;

%check objects
if ~isfield(ld,'objects'), ld.objects = []; end;


end

function write(fname, ld) %#ok<INUSD>
% WRITE saves label data to a file
%
% INPUTS
% ------
% fname   - the input file name
% ld      - the input label data
% 
% OUTPUTS
% -------
%

%load the file
save(fname, 'ld', '-mat');

end

function [obj] = createObj(objId)
% CREATEOBJ creates a new object and returns it
%
% INPUTS
% ------
% objId   - the object id of the new object
%  
% OUTPUTS
% -------
% obj     - the new obj
%

obj = struct('id', objId);

end

function [ld, objId] = addObj(ld)
% ADDOBJ adds a new object and returns the object id
%
% INPUTS
% ------
% ld      - the input label data
%  
% OUTPUTS
% -------
% ld      - the output label data
% objId   - the id of the new object added
%

%get id of new object
objId = max([ld.objects.id]) + 1;
if isempty(objId), objId = 1; end;
%add it
ld.objects = [ld.objects createObj(objId)];

end

function ld = removeObj(ld, objId)
% REMOVEOBJ deletes an object and clears objects of every label with that
% object id
%
% INPUTS
% ------
% ld      - the input label data
% objId   - the id of the object to remove
%  
% OUTPUTS
% -------
% ld      - the output label data
%

%get index of object
objInd = find([ld.objects.id] == objId);
%make sure it's valid
if ~isempty(objInd)
  %clear it
  ld.objects(objInd) = [];
  %update all labels with that object id, loop all frames and check
  for f=1:length(ld.frames)
    %reset labels with that object label
    for l=1:length(ld.frames(f).labels) 
      if ld.frames(f).labels(l).obj == objId, 
        ld.frames(f).labels(l).obj = []; 
      end; 
    end;
%     lbls = find([ld.frames(f).labels.obj] == objId);
%     for l=lbls, ld.frames(f).labels(l).obj = []; end;
  end;
end; %if  

end

function [objIds] = getObjIds(ld)
% GETOBJIDS returns the object ids present
%
% INPUTS
% ------
% ld      - the input label data
%  
% OUTPUTS
% -------
% objIds  - the list of object ids
%

%get ids of objects
objIds = [ld.objects.id];

end

function nframes = nFrames(ld)
% NFRAMES returns the number of frames
%
% INPUTS
% ------
% ld        - the input label data
% 
% OUTPUTS
% -------
% nframes   - the number of frames
%

%get the frame
nframes = length(ld.frames);
end

function frame = getFrame(ld, frameIdx)
% GETFRAME returns the required frame
%
% INPUTS
% ------
% ld        - the input label data
% frameIdx  - the frame index
% 
% OUTPUTS
% -------
% frame     - the returned frame, which is a structure with fields
%             .frame    - the index or file name of the frame
%             .labels   - the array of labels in this frame
%

%get the frame
frame = ld.frames(frameIdx);
end

function frm = createFrame(frame, labels)
% CREATEFRAME creates a new frame
%
% INPUTS
% ------
% frame     - the frame id or file name
% labels    - the frame labels
% 
% OUTPUTS
% -------
% frm     - the output new frame
%
if nargin<1, frame = [];  end;
if nargin<2, labels = createLabel(); end;

%create the new frame
frm = struct('frame',frame, 'labels',labels);

end

function [ld, frameIdx] = addFrame(ld, frame, labels)
% ADDFRAME adds a frame into the data structure
%
% INPUTS
% ------
% ld        - the input label data
% frame     - the frame id or file name
% labels    - the frame labels
% 
% OUTPUTS
% -------
% ld        - the update ld structure
% frameIdx  - the index of the new frame
%
if nargin<2, frame = [];  end;
if nargin<3, labels = createLabel(); end;

%get the frame index
frameIdx = length(ld.frames) + 1;

%put the new frame
ld.frames(frameIdx) = createFrame(frame, labels);

end

function ld = removeFrame(ld, frameIdx)
% REMOVEFRAME removes the frame
%
% INPUTS
% ------
% ld        - the input label data
% frameIdx  - the frame index
% 
% OUTPUTS
% -------
% ld        - the update ld structure
%

%remove the frame
ld.frames(frameIdx) = [];
end

function label = createLabel(points, type, subtype, objId)
% CREATELABEL creates a new label
%
% INPUTS
% ------
% points    - the points for that label
% type      - the type of label
% subtype   - the subtype of the label
% objId     - the objId of the label
% 
% OUTPUTS
% -------
% ld      - the output updated label data
%
if nargin<1,  points = {};  end;
if nargin<2,  type = [];    end;
if nargin<3,  subtype = []; end;
if nargin<4,  objId = [];   end;

%create a new label
label = struct('points',points, 'type',type, ...
  'subtype',subtype, 'obj',objId);

end

function nl = nLabels(ld, frameIdx)
% NLABELS gets the number of labels in the required frame
%
% INPUTS
% ------
% ld        - the input label data
% frameIdx  - the frame index
% 
% OUTPUTS
% -------
% nl        - the number of labels
%

nl = length(ld.frames(frameIdx).labels);

end


function [ld, lblIdx] = addLabel(ld, frameIdx, points, type, subtype, objId)
% ADDLABEL adds a new label
%
% INPUTS
% ------
% ld        - the input label data
% frameIdx  - the frame index
% points    - the points for that label or the label structure if given
% type      - the type of label
% subtype   - the subtype of the label
% objId     - the objId of the label
% 
% OUTPUTS
% -------
% ld      - the output updated label data
% lblIdx  - the new label index
%
if nargin<3,  points = [];  end;
if nargin<4,  type = [];    end;
if nargin<5,  subtype = []; end;
if nargin<6,  objId = [];   end;

%get the new label index
lblIdx = nLabels(ld, frameIdx) + 1;

%create the new label if not a struct
if isstruct(points),  label = points;
else                  label = createLabel(points, type, subtype, objId);
end;

%add the label to the required frame
ld.frames(frameIdx).labels(lblIdx) = label;

end

function ld = updateLabel(ld, frameIdx, lblIdx, points, type, subtype, objId)
% UPDATELABEL updates an existing label
%
% INPUTS
% ------
% ld        - the input label data
% frameIdx  - the frame id
% lblIdx    - the index of the label to change
% points    - the points for that label (don't change if nan). It can also
%             be a structure, in which case it is a label structure, 
%             so just replace it
% type      - the type of label (don't change if nan)
% subtype   - the subtype of the label (don't change if nan)
% objId     - the objId of the label (don't change if nan)
% 
% OUTPUTS
% -------
% ld      - the output updated label data
%

%check if just to replace it
if nargin>=4 && isstruct(points)
  label = points;
%we are passaed in independent components of the labels
else
  %get the label
  label = ld.frames(frameIdx).labels(lblIdx);

  %update the label
  if nargin>=7 && ~any(isnan(objId)),   label.obj = objId;        end;
  if nargin>=6 && ~any(isnan(subtype)), label.subtype = subtype;  end;
  if nargin>=5 && ~any(isnan(type)),    label.type = type ;       end;
  if nargin>=4 && ~any(any(isnan(points))),  label.points = points;    end;
end;

%put it back
ld.frames(frameIdx).labels(lblIdx) = label;
end

function label = getLabel(ld, frameIdx, lblIdx)
% GETLABEL retuns the required label
%
% INPUTS
% ------
% ld        - the input label data
% frameIdx  - the frame index
% lblIdx    - the index of the label to return. If empty or absent, then 
%             return the labels in this frame
% 
% OUTPUTS
% -------
% label     - the returned label(s), which is a structure with fields
%             .points   - the label points
%             .type     - the label type
%             .subtype  - the label subtype
%             .obj      - the label object id
%

%get the label
if nargin<3 || isempty(lblIdx)
  lblIdx = 1:length(ld.frames(frameIdx).labels);
end;
%return
label = ld.frames(frameIdx).labels(lblIdx);

end

function ld = removeLabel(ld, frameIdx, lblIdx)
% REMOVELABEL removes a label
%
% INPUTS
% ------
% ld        - the input label data
% frameIdx  - the frame id
% lblIdx    - the index of the label to remove
% 
% OUTPUTS
% -------
% ld      - the output updated label data
%

%remove the label
ld.frames(frameIdx).labels(lblIdx) = [];

end


