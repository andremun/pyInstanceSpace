function pyis_export_geometry_cases(toolkit, destination)
% Controlled stage-local geometry references, independent of fitted pipelines.
% SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
assert(~isfolder(destination), 'Destination must be new.');
addpath(fullfile(toolkit,'core'), fullfile(toolkit,'utils'), fullfile(toolkit,'output'));
[status,commit] = system(sprintf('git -C "%s" rev-parse HEAD', toolkit));
assert(status == 0);
[status,dirty] = system(sprintf('git -C "%s" status --porcelain', toolkit));
assert(status == 0 && isempty(strtrim(dirty)), 'MATLAB checkout must be clean.');
mkdir(destination);
defaults = ISAdefaults(struct());
opts = defaults.cloister;
opts.pval = 0.05; opts.corrThreshold = 0.7; opts.maxFeatures = 20;
[u,v,w] = ndgrid([-1 1],[-1 1],[-1 1]);
cube = [u(:),v(:),w(:)];
[u,v] = ndgrid([-1 1],[-1 1]);
square = [u(:),v(:)];
inputs = {cube, square, cube, cube};
projections = {eye(3), [1 0;0 1;1 2], [1 0 0;0 0 0;0 0 0], eye(3)};
names = {'solid','coplanar','collinear','fallback'};
for i = 1:numel(names)
    options = opts;
    if strcmp(names{i},'fallback'), options.maxFeatures = 2; end
    item = struct('x',inputs{i},'a',projections{i},'options',options);
    try
        result = CLOISTER(item.x,item.a,options);
        item.vertices = result.Zedge; item.faces = result.ZedgeFaces;
        item.correlated_vertices = result.Zecorr;
        item.correlated_faces = result.ZecorrFaces;
        item.error = '';
    catch err
        item.error = err.identifier;
    end
    writejson(fullfile(destination,['cloister_' names{i} '.json']),item);
end
[u,v] = ndgrid(-3:3,-3:3);
points = [u(:),v(:)];
ring = points(max(abs(points),[],2)>=2,:);
[u,v] = ndgrid(0:2,0:2); square = [u(:),v(:)];
supports = {ring, [square;square+[6 0]]};
queries = {[0 0;2.5 0;3 0;4 0;0 2.5], [1 1;7 1;4 1;0 0;9 0]};
names = {'hole','components'};
traceOpts = defaults.trace;
traceOpts.method = 'trace3'; traceOpts.parallel = false;
traceOpts.PI = 0.9; traceOpts.minAreaFrac = 0; traceOpts.minInstances = 4;
scriptfcn; % Toolkit boundary tracing, including every hole cycle.
for i = 1:numel(names)
    support = supports{i}; query = queries{i};
    fixed = alphaShape(support,0.8);
    Z = [support;query]; labels = [true(size(support,1),1);false(size(query,1),1)];
    result = TRACE(Z,labels,[],ones(size(Z,1),1),labels,{'algorithm'},traceOpts);
    footprint = result.good{1};
    item = struct('support',support,'queries',query,'alpha',0.8,...
        'fixed_area',area(fixed),'fixed_regions',numRegions(fixed),...
        'fixed_membership',inShape(fixed,query),'fixed_boundary',traceAlphaBoundary(fixed),...
        'z',Z,'labels',labels,'options',traceOpts,...
        'trace_area',footprint.measure,'trace_elements',footprint.elements,...
        'trace_good_elements',footprint.goodElements,'trace_purity',footprint.purity,...
        'trace_regions',numRegions(footprint.polygon),...
        'trace_membership',inShape(footprint.polygon,query),...
        'trace_boundary',traceAlphaBoundary(footprint.polygon));
    writejson(fullfile(destination,['trace_' names{i} '.json']),item);
end
files = dir(fullfile(destination,'*.json'));
entries = struct('path',{},'sha256',{});
for i=1:numel(files)
    entries(i).path = files(i).name;
    entries(i).sha256 = filehash(fullfile(destination,files(i).name));
end
manifest = struct('schema_version','pyinstancespace.geometry-reference/v1',...
    'matlab_commit',strtrim(commit),'matlab_release',version('-release'),...
    'matlab_version',version,'platform',computer,'toolboxes',ver,...
    'generator','tests/matlab_export/pyis_export_geometry_cases.m',...
    'generator_sha256',filehash([mfilename('fullpath') '.m']),...
    'files',entries);
writejson(fullfile(destination,'manifest.json'),manifest);
end

function writejson(path,value)
fid=fopen(path,'w'); guard=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end

function value=filehash(path)
fid=fopen(path,'r'); guard=onCleanup(@() fclose(fid)); %#ok<NASGU>
bytes=fread(fid,Inf,'*uint8');
digest=java.security.MessageDigest.getInstance('SHA-256');
digest.update(bytes);
value=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[]));
end
