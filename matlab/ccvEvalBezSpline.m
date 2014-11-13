function [outPoints, tangent] = ccvEvalBezSpline(spline, h)
% ccvEvalBezSpline evaluates a Bezier spline with the specified degree

% INPUTS
% ------
% spline    - input 3 or 4 points in a matrix 3x2 or 4x2 [xs, ys]
% h         - [0.05] the interval to use for evaluation
% 
% OUTPUTS
% -------
% outPoints - output points nx2 [xs; ys]
% tangent   - the tangent at the two end-points [t0; t1]
%
% EXAMPLE
% -------
% [p, t] = ccvEvalBezSpline(sp, 0.1);
% 
% See also ccvDrawBezSpline

if nargin<2, h = 0.05; end

%get the degree
degree = size(spline, 1) - 1;

%compute number of return points
n = floor(1/h)+1;

%degree
switch degree
	%Quadratic Bezier curve
	case 2
		M =	[1,	-2,	1; ...
			-2,	2,	0; ...
			1,	0,	0];

		%compute constants [a, b, c]
		abcd = M * spline;
		a = abcd(1,:);	b = abcd(2,:);	c = abcd(3,:);

		%compute at time 0
		P = c;
		dP = b * h + a * h^2;
		ddP = 2*a*h^2;

		%loop
		outPoints = zeros(n, size(spline,2));
		outPoints(1,:) = P;
		for i=2:n
			%calculate new point
			P = P + dP;
			%update steps
			dP = dP + ddP;
			%put back
			outPoints(i,:) = P;
		end;
		
		%tangents: t0 = b
		t0 = b;
		%t1 = 2a+b
		t1 = 2*a+b;

	%Cubic Bezier curve
	case 3
		M =	[-1,	3,		-3,		1; ...
			3,		-6,		3,		0; ...
			-3,		3,		0,		0; ...
			1,		0,		0,		0];

		%compute constants [a, b, c, d]
		abcd = M * spline;
		a = abcd(1,:);	b = abcd(2,:);	c = abcd(3,:);	d = abcd(4,:);

		%compute at time 0
		P = d;
		dP = c*h + b * h^2 + a * h^3;
		ddP = 2*b*h^2 + 6*a*h^3;
		dddP = 6*a*h^3;

		%loop
		outPoints = zeros(n, size(spline,2));
		outPoints(1,:) = P;
		for i=2:n
			%calculate new point
			P = P + dP;
			%update steps
			dP = dP + ddP;
			ddP = ddP + dddP;
			%put back
			outPoints(i,:) = P;
		end;
		
		%tangents: t0 = c
		t0 = c;
		%t1 = 3a+2*b+c
		t1 = 3*a+2*b+c;
end;

%put tangents together
tangent = [t0; t1];

