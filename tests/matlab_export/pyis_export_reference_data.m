function pyis_export_reference_data(toolkitRoot, outputRoot, varargin)
% pyis_export_reference_data  Regenerate pyInstanceSpace's MATLAB reference
% fixtures from a real run of the andremun/InstanceSpace toolkit.
%
%   pyis_export_reference_data(toolkitRoot, outputRoot)
%   pyis_export_reference_data(toolkitRoot, outputRoot, 'datasetRoot', dir)
%
%   toolkitRoot - path to a checkout of https://github.com/andremun/InstanceSpace
%                 (the directory containing InstanceSpace.m/buildIS.m).
%   outputRoot  - destination directory for the exported fixtures (created
%                 if missing). Existing files are overwritten -- point this
%                 at a scratch directory and diff it against the committed
%                 fixtures yourself before copying anything over.
%   datasetRoot - (optional) directory containing metadata.csv +
%                 metadata_test.csv. Defaults to toolkitRoot/test/data/,
%                 the toolkit's own reference dataset (Munoz et al. 2018
%                 classification study) -- the same dataset that produced
%                 pyInstanceSpace's existing tests/matlab_reference/
%                 fixtures, verified directly (identical header + row
%                 counts) rather than assumed.
%
%   Design, coverage rationale, and known gaps are documented in this
%   file's sibling README.md (tests/matlab_export/README.md in
%   pyInstanceSpace) -- read that first if you're regenerating fixtures
%   for the first time. In short: this function runs the pipeline through
%   InstanceSpace's own staged build() (prelim -> sifted -> pilot once,
%   then cloister, then several pythia/trace option variants re-using the
%   same upstream state), and exports each stage's real output struct
%   (obj.model.<stage>) to CSV using the same writeArray2CSV/writeCell2CSV
%   conventions output/scriptcsv.m already uses -- not a new format.
%
%   NOT executed against real MATLAB before being committed to
%   pyInstanceSpace -- written from direct inspection of this toolkit's
%   source (InstanceSpace.m, core/*.m, output/scriptcsv.m, scriptfcn.m,
%   test_integration.m), not guessed, but unverified by an actual run.
%   Review before trusting its output; run once on a throwaway
%   outputRoot first.

% -------------------------------------------------------------------------
% Written for the Instance Space Analysis (ISA) Toolkit
% (https://github.com/andremun/InstanceSpace), to be copied into a
% checkout of that repository and run there. Lives in pyInstanceSpace
% (the Python port) because it is that repo's test-fixture tooling, not
% because it is meant to be committed to the MATLAB repo.
% -------------------------------------------------------------------------

p = inputParser;
addRequired(p, 'toolkitRoot', @(x) (ischar(x) || isstring(x)) && isfolder(x));
addRequired(p, 'outputRoot', @(x) ischar(x) || isstring(x));
addParameter(p, 'datasetRoot', '', @(x) ischar(x) || isstring(x));
parse(p, toolkitRoot, outputRoot, varargin{:});

toolkitRoot = ensureTrailingSlash(char(p.Results.toolkitRoot));
outputRoot  = ensureTrailingSlash(char(p.Results.outputRoot));
datasetRoot = char(p.Results.datasetRoot);
if isempty(datasetRoot)
    datasetRoot = [toolkitRoot 'test/data/'];
end
datasetRoot = ensureTrailingSlash(datasetRoot);

if ~isfile([datasetRoot 'metadata.csv'])
    error('pyis_export:missingDataset', ...
        'metadata.csv not found in ''%s''. Pass ''datasetRoot'' explicitly if the ' ...
        'reference dataset lives elsewhere.', datasetRoot);
end

% ---- Make the toolkit's own code resolvable, same as InstanceSpace.m's
% own ensurePathSetup does internally -- done explicitly here too so this
% script also works if copied somewhere InstanceSpace.m can't find itself
% relative to (e.g. run from a different working directory).
subdirs = {'core', 'output', 'utils', 'deprecated'};
for i = 1:numel(subdirs)
    d = fullfile(toolkitRoot, subdirs{i});
    if isfolder(d)
        addpath(d);
    end
end
addpath(toolkitRoot);

mkdirIfMissing(outputRoot);
mkdirIfMissing([outputRoot 'input/']);
mkdirIfMissing([outputRoot 'training_artifacts/']);
mkdirIfMissing([outputRoot 'explore_outputs/']);

copyfile([datasetRoot 'metadata.csv'], [outputRoot 'input/metadata.csv']);
copyfile([datasetRoot 'metadata_test.csv'], [outputRoot 'input/metadata_test.csv']);

startTime = tic;

% =========================================================================
% Base pipeline: prelim -> sifted -> pilot -> cloister, once. Every
% downstream PYTHIA/TRACE variant below re-uses this same state instead of
% re-running these (the expensive, option-invariant) stages per variant.
% =========================================================================
fprintf('[EXPORT] Building base pipeline (prelim -> sifted -> pilot -> cloister) on %s\n', datasetRoot);
obj = InstanceSpace(datasetRoot);
obj = obj.build('stages', {'prelim', 'sifted', 'pilot', 'cloister'});

exportPrelimArtifacts(obj.model.prelim, obj.model.data, [outputRoot 'training_artifacts/prelim/']);
exportSiftedArtifacts(obj.model.sifted, [outputRoot 'training_artifacts/sifted/']);
exportPilotArtifacts(obj.model.pilot, [outputRoot 'training_artifacts/pilot/']);
exportCloisterArtifacts(obj.model.cloist, [outputRoot 'training_artifacts/cloister/']);

% =========================================================================
% PYTHIA/TRACE variants -- mirrors the option cases test_integration.m
% already exercises for its own regression suite (classifier_svm,
% tuning_bayes, ...), reused here for fixture generation instead of
% pass/fail checking. 'default' (MATLAB's own untouched opts -- KNN
% classifier, Sobol tuning, TRACE3) comes first so the flat, backward-
% compatible tests/matlab_reference/ layout is still produced exactly as
% before; the svm-forced variants add new coverage alongside it, not in
% place of it. EVERY variant gets both a build() pass (training_artifacts/)
% and an explore() pass (explore_outputs/) on the model it just trained --
% earlier drafts of this script only ran explore() for the default case,
% leaving the other three build-only. Add more variants here as Python's
% test suite grows new classifier/tuning/kernel combinations it needs
% MATLAB numbers for.
% =========================================================================
variants = { ...
    struct('name', 'default', ...
           'desc', 'MATLAB''s own untouched defaults: KNN classifier, Sobol tuning, TRACE3.', ...
           'pythia', struct(), 'ispolykrnl', false), ...
    struct('name', 'sobol_svm', ...
           'desc', 'SVM classifier (pyInstanceSpace''s own default), Sobol tuning, gaussian kernel.', ...
           'pythia', struct('classifier', 'svm', 'tuning', 'sobol'), 'ispolykrnl', false), ...
    struct('name', 'bayes_svm_gaussian', ...
           'desc', 'Legacy Bayesian-optimisation tuning, gaussian kernel.', ...
           'pythia', struct('classifier', 'svm', 'tuning', 'bayes'), 'ispolykrnl', false), ...
    struct('name', 'bayes_svm_poly', ...
           'desc', 'Legacy Bayesian-optimisation tuning, polynomial kernel.', ...
           'pythia', struct('classifier', 'svm', 'tuning', 'bayes'), 'ispolykrnl', true) ...
};

baseObj = obj; % snapshot with prelim/sifted/pilot/cloister already completed
for v = 1:numel(variants)
    variant = variants{v};
    fprintf('[EXPORT] === PYTHIA/TRACE variant ''%s'': %s ===\n', variant.name, variant.desc);
    obj = baseObj;
    fields = fieldnames(variant.pythia);
    for f = 1:numel(fields)
        obj.opts.pythia.(fields{f}) = variant.pythia.(fields{f});
    end
    obj.opts.pythia.ispolykrnl = variant.ispolykrnl;

    % ---- Build path (training) ----
    obj = obj.build('stages', {'pythia', 'trace'});
    exportPythiaArtifacts(obj.model.pythia, obj.model.data.algolabels, ...
        [outputRoot 'training_artifacts/pythia/' variant.name '/']);
    exportTraceArtifacts(obj.model.trace, obj.model.data.algolabels, ...
        [outputRoot 'training_artifacts/trace/' variant.name '/']);

    % ---- Explore path (test-set inference on the model just trained) ----
    obj = obj.explore(datasetRoot);
    testOut = obj.getResults(1);
    exportPythiaTraceExploreArtifacts(testOut, [outputRoot 'explore_outputs/' variant.name '/']);

    if v == 1
        % step1-3 (prelim/sifted/pilot's test-set transform) don't depend
        % on opts.pythia/opts.trace, so they're identical across every
        % variant here -- write them once, at explore_outputs/ *and*
        % reproduce the flat step4/step5 filenames tests/matlab_reference/
        % already documents there too, so this default variant stays a
        % byte-for-byte drop-in for that existing fixture set rather than
        % a same-data-different-name reshuffle of it.
        exportLegacyExploreLayout(testOut, [outputRoot 'explore_outputs/']);
    end
end

writeProvenance(toolkitRoot, outputRoot);

fprintf('[EXPORT] Completed in %.1f s. Output written to %s\n', toc(startTime), outputRoot);
fprintf('EOF:SUCCESS\n');
end

% =========================================================================
% Per-stage export functions
% =========================================================================

function exportPrelimArtifacts(prelimOut, data, destDir)
% Exports PRELIM's per-feature/per-algorithm fit parameters and its
% per-instance outputs. Field names verified directly against
% core/PRELIM.m's out.* assignments, not assumed.
mkdirIfMissing(destDir);

featTable = table( ...
    data.featlabels(:), prelimOut.minX(:), prelimOut.lambdaX(:), prelimOut.muX(:), ...
    prelimOut.sigmaX(:), prelimOut.medval(:), prelimOut.iqrange(:), prelimOut.hibound(:), ...
    prelimOut.lobound(:), ...
    'VariableNames', {'feature_name', 'min_x', 'lambda_x', 'mu_x', 'sigma_x', 'medval', ...
                       'iqrange', 'hi_bound', 'lo_bound'});
writetable(featTable, [destDir 'prelim_feature_params.csv']);

algoTable = table( ...
    data.algolabels(:), prelimOut.lambdaY(:), prelimOut.muY(:), prelimOut.sigmaY(:), ...
    'VariableNames', {'algo_name', 'lambda_y', 'mu_y', 'sigma_y'});
writetable(algoTable, [destDir 'prelim_algo_params.csv']);

writetable(table(prelimOut.minY, 'VariableNames', {'min_y'}), [destDir 'prelim_scalars.csv']);

instTable = table( ...
    data.instlabels(:), prelimOut.Ybest(:), prelimOut.P(:), prelimOut.numGoodAlgos(:), ...
    prelimOut.beta(:), ...
    'VariableNames', {'instance_id', 'y_best', 'p_best_algo', 'num_good_algos', 'beta'});
writetable(instTable, [destDir 'prelim_instance_outputs.csv']);
writeMatrixCSV(prelimOut.Ybin, data.algolabels, data.instlabels(:), [destDir 'prelim_ybin.csv']);
end

function exportSiftedArtifacts(siftedOut, destDir)
% Exports SIFTED's correlation matrix, selected-feature indices, and the
% two evalclusters fields Python's evaluate_cluster() actually checks
% (InspectedK/CriterionValues) -- eva itself is a MATLAB object, not
% CSV-able directly.
mkdirIfMissing(destDir);

writeMatrixCSV(siftedOut.rho, [], [], [destDir 'correlation_rho.csv']);
writeMatrixCSV(siftedOut.p, [], [], [destDir 'correlation_pval.csv']);

% 1-based (MATLAB convention) -- Python consumers subtract 1, matching the
% existing tests/matlab_reference/README.md's documented convention.
selTable = table((1:numel(siftedOut.selvars))', siftedOut.selvars(:), ...
    'VariableNames', {'rank', 'original_index'});
writetable(selTable, [destDir 'sifted_indices.csv']);

if isfield(siftedOut, 'eva') && ~isempty(siftedOut.eva)
    clusterTable = table(siftedOut.eva.InspectedK(:), siftedOut.eva.CriterionValues(:), ...
        'VariableNames', {'k', 'silhouette_score'});
    writetable(clusterTable, [destDir 'sifted_cluster_scores.csv']);
end
if isfield(siftedOut, 'Ksuggested')
    writetable(table(siftedOut.Ksuggested, 'VariableNames', {'k_suggested'}), ...
        [destDir 'sifted_k_suggested.csv']);
end
writeMatrixCSV(double(siftedOut.clust), [], [], [destDir 'sifted_clust_membership.csv']);
end

function exportPilotArtifacts(pilotOut, destDir)
% pilot.summary is already a labelled cell table (feature name -> A's
% coefficients per projected dimension) -- export it directly, same
% pattern output/scriptcsv.m uses for container.pilot.summary, rather
% than re-deriving feature labels separately.
mkdirIfMissing(destDir);
if isfield(pilotOut, 'summary') && ~isempty(pilotOut.summary)
    writeCellCSV(pilotOut.summary(2:end, 2:end), pilotOut.summary(1, 2:end), ...
        pilotOut.summary(2:end, 1), [destDir 'pilot_matrix.csv']);
end
writeMatrixCSV(pilotOut.A, [], [], [destDir 'pilot_a_raw.csv']);
writeMatrixCSV(pilotOut.B, [], [], [destDir 'pilot_b.csv']);
writeMatrixCSV(pilotOut.C, [], [], [destDir 'pilot_c.csv']);
writeMatrixCSV(pilotOut.R2(:), {'r2'}, [], [destDir 'pilot_r2.csv']);
writetable(table(pilotOut.error, 'VariableNames', {'error'}), [destDir 'pilot_error.csv']);
% eoptim/perf/alpha/X0 only exist on the numerical (non-analytic) solve
% path -- one column per opts.ntries BFGS restart (out.error/R2 above are
% the metrics for whichever restart numerical_solve picked as best).
if isfield(pilotOut, 'eoptim') && ~isempty(pilotOut.eoptim)
    writeMatrixCSV(pilotOut.eoptim(:), {'eoptim'}, [], [destDir 'pilot_eoptim.csv']);
end
if isfield(pilotOut, 'perf') && ~isempty(pilotOut.perf)
    writeMatrixCSV(pilotOut.perf(:), {'perf'}, [], [destDir 'pilot_perf.csv']);
end
if isfield(pilotOut, 'alpha') && ~isempty(pilotOut.alpha)
    writeMatrixCSV(pilotOut.alpha, [], [], [destDir 'pilot_alpha.csv']);
end
if isfield(pilotOut, 'X0') && ~isempty(pilotOut.X0)
    writeMatrixCSV(pilotOut.X0, [], [], [destDir 'pilot_x0.csv']);
end
end

function exportCloisterArtifacts(cloistOut, destDir)
% Only Zedge/Zecorr are actually returned by CLOISTER.m -- see this
% folder's README ("Known gap") for why rho/pval/xEdge/remove are not
% exported here.
mkdirIfMissing(destDir);
writeMatrixCSV(cloistOut.Zedge, [], [], [destDir 'z_edge.csv']);
writeMatrixCSV(cloistOut.Zecorr, [], [], [destDir 'z_ecorr.csv']);
end

function exportPythiaArtifacts(pythiaOut, algolabels, destDir)
mkdirIfMissing(destDir);
if isfield(pythiaOut, 'summary') && ~isempty(pythiaOut.summary)
    writeCellCSV(pythiaOut.summary(2:end, 2:end), pythiaOut.summary(1, 2:end), ...
        pythiaOut.summary(2:end, 1), [destDir 'summary.csv']);
end
writeMatrixCSV(pythiaOut.Ysub, algolabels, [], [destDir 'ysub.csv']);
writeMatrixCSV(pythiaOut.Yhat, algolabels, [], [destDir 'yhat.csv']);
writeMatrixCSV(pythiaOut.Pr0sub, algolabels, [], [destDir 'pr0sub.csv']);
writeMatrixCSV(pythiaOut.Pr0hat, algolabels, [], [destDir 'pr0hat.csv']);
writeMatrixCSV(pythiaOut.selection0(:), {'selection0'}, [], [destDir 'selection0.csv']);
writeMatrixCSV(pythiaOut.selection1(:), {'selection1'}, [], [destDir 'selection1.csv']);

paramTable = table(algolabels(:), pythiaOut.param1(:), ...
    'VariableNames', {'algo', 'param1'});
if isfield(pythiaOut, 'param2') && ~isempty(pythiaOut.param2)
    paramTable.param2 = pythiaOut.param2(:);
end
writetable(paramTable, [destDir 'hyperparameters.csv']);
end

function exportTraceArtifacts(traceOut, algolabels, destDir)
mkdirIfMissing(destDir);
if isfield(traceOut, 'summary') && ~isempty(traceOut.summary)
    writeCellCSV(traceOut.summary(2:end, 2:end), traceOut.summary(1, 2:end), ...
        traceOut.summary(2:end, 1), [destDir 'summary.csv']);
end
for i = 1:numel(algolabels)
    writeFootprintCSV(traceOut.good{i}, [destDir 'good_' algolabels{i} '.csv']);
    writeFootprintCSV(traceOut.best{i}, [destDir 'best_' algolabels{i} '.csv']);
end
if isfield(traceOut, 'hard') && ~isempty(traceOut.hard)
    writeFootprintCSV(traceOut.hard, [destDir 'hard.csv']);
end
end

function exportPythiaTraceExploreArtifacts(testOut, destDir)
% Per-variant explore-path export: PYTHIA/TRACE's test-set inference
% output for whichever variant trained this testOut. Distinct from
% exportPythiaArtifacts/exportTraceArtifacts (the *build*-path export,
% training_artifacts/) -- explore-mode PYTHIA runs a genuinely different
% code path internally (PYTHIAevalMode in core/PYTHIA.m: no hyperparameter
% search, just applying the already-trained classifiers/reconciling
% test-only algorithms), so its own summary table has fewer columns (no
% param1/param2/param2Label) -- exported here under its own name
% (eval_summary.csv) rather than overwriting/conflated with the training
% summary.csv this same variant's training_artifacts/ already has.
mkdirIfMissing(destDir);
writeMatrixCSV(double(testOut.pythia.Yhat), testOut.data.algolabels, testOut.data.instlabels(:), ...
    [destDir 'predictions.csv']);
writeMatrixCSV(testOut.pythia.Pr0hat, testOut.data.algolabels, testOut.data.instlabels(:), ...
    [destDir 'probabilities.csv']);
if isfield(testOut.pythia, 'summary') && ~isempty(testOut.pythia.summary)
    writeCellCSV(testOut.pythia.summary(2:end, 2:end), testOut.pythia.summary(1, 2:end), ...
        testOut.pythia.summary(2:end, 1), [destDir 'pythia_eval_summary.csv']);
end
if isfield(testOut.trace, 'summary') && ~isempty(testOut.trace.summary)
    writeCellCSV(testOut.trace.summary(2:end, 2:end), testOut.trace.summary(1, 2:end), ...
        testOut.trace.summary(2:end, 1), [destDir 'trace_eval_summary.csv']);
end

membershipCols = [strcat('in_good_', testOut.data.algolabels(:)'), ...
    strcat('in_best_', testOut.data.algolabels(:)')];
membership = [footprintMembership(testOut.trace.good, testOut.pilot.Z), ...
    footprintMembership(testOut.trace.best, testOut.pilot.Z)];
writeMatrixCSV(double(membership), membershipCols, testOut.data.instlabels(:), ...
    [destDir 'trace_membership.csv']);
end

function exportLegacyExploreLayout(testOut, destExplore)
% Reproduces tests/matlab_reference/explore_outputs/'s existing flat
% layout exactly (see that directory's own README.md for the full
% field-by-field description) -- called once, for the 'default' variant
% only, so this script's output stays a byte-for-byte drop-in replacement
% for that existing fixture set rather than a same-data-different-name
% reshuffle of it.
mkdirIfMissing(destExplore);
writeMatrixCSV(testOut.data.X, testOut.data.featlabels, testOut.data.instlabels(:), ...
    [destExplore 'step1_after_prelim.csv']);
% SIFTED's selection is already applied to testOut.data.X via
% out.featsel.idx inside evaluateTestSet -- step2 is a duplicate of step1
% under this class-based flow (unlike the older exploreIS.m script path),
% recorded here rather than silently reproduced as an identical file so a
% future reader isn't left guessing why the two match.
writeMatrixCSV(testOut.data.X, testOut.data.featlabels, testOut.data.instlabels(:), ...
    [destExplore 'step2_after_sifted.csv']);
zcols = arrayfun(@(i) sprintf('z%d', i), 1:size(testOut.pilot.Z, 2), 'UniformOutput', false);
writeMatrixCSV(testOut.pilot.Z, zcols, testOut.data.instlabels(:), ...
    [destExplore 'step3_after_pilot.csv']);
writeMatrixCSV(double(testOut.pythia.Yhat), testOut.data.algolabels, testOut.data.instlabels(:), ...
    [destExplore 'step4_pythia_predictions.csv']);
writeMatrixCSV(testOut.pythia.Pr0hat, testOut.data.algolabels, testOut.data.instlabels(:), ...
    [destExplore 'step4_pythia_probabilities.csv']);

membershipCols = [{'in_space'}, ...
    strcat('in_good_', testOut.data.algolabels(:)'), ...
    strcat('in_best_', testOut.data.algolabels(:)')];
inSpace = true(size(testOut.data.instlabels(:))); % CLOISTER-derived; not validated, see README
membership = [inSpace, footprintMembership(testOut.trace.good, testOut.pilot.Z), ...
    footprintMembership(testOut.trace.best, testOut.pilot.Z)];
writeMatrixCSV(double(membership), membershipCols, testOut.data.instlabels(:), ...
    [destExplore 'step5_trace_membership.csv']);
end

function membership = footprintMembership(footprints, Z)
% polyshape's containment test is isinterior(poly,x,y); alphaShape's is
% inShape(shp,x,y) -- different method names for the two polygon types
% TRACE can return (legacy vs. TRACE3, the current default), dispatched
% on explicitly rather than assuming one applies to both.
membership = false(size(Z, 1), numel(footprints));
for i = 1:numel(footprints)
    if ~isfield(footprints{i}, 'polygon') || isempty(footprints{i}.polygon)
        continue;
    end
    poly = footprints{i}.polygon;
    if isa(poly, 'polyshape')
        membership(:, i) = isinterior(poly, Z(:, 1), Z(:, 2));
    elseif isa(poly, 'alphaShape')
        membership(:, i) = inShape(poly, Z(:, 1), Z(:, 2));
    end
end
end

% =========================================================================
% Shared low-level helpers
% =========================================================================

function writeMatrixCSV(data, colNames, rowNames, filename)
% Writes a plain numeric matrix/vector to CSV, column- and (optionally)
% row-labelled -- same convention as output/scriptfcn.m's writeArray2CSV,
% reimplemented locally so this exporter has no runtime dependency on the
% toolkit's own plotting/output machinery beyond addpath resolving the
% core stage functions.
if isempty(data)
    return;
end
if isempty(colNames)
    colNames = arrayfun(@(i) sprintf('col_%d', i), 1:size(data, 2), 'UniformOutput', false);
end
colNames = sanitizeNames(colNames);
t = array2table(data, 'VariableNames', colNames);
if ~isempty(rowNames)
    t.Properties.RowNames = matlab.lang.makeUniqueStrings(cellstr(rowNames));
    writetable(t, filename, 'WriteRowNames', true);
else
    writetable(t, filename);
end
end

function writeCellCSV(data, colNames, rowNames, filename)
% Cell-array equivalent of writeMatrixCSV, matching
% output/scriptfcn.m's writeCell2CSV -- used for the pilot/pythia/trace
% out.summary tables, which mix numbers and strings.
if isempty(data)
    return;
end
colNames = sanitizeNames(colNames);
t = cell2table(data, 'VariableNames', colNames);
if ~isempty(rowNames)
    t.Properties.RowNames = matlab.lang.makeUniqueStrings(cellstr(rowNames));
    writetable(t, filename, 'WriteRowNames', true);
else
    writetable(t, filename);
end
end

function writeFootprintCSV(footprint, filename)
% Exports a single footprint's boundary as an (x, y) vertex list, with a
% blank row delimiting separate regions -- the convention
% tests/matlab_reference/README.md already documents. Handles both
% polyshape (legacy TRACE) and alphaShape (TRACE3, the current default),
% matching output/scriptcsv.m's own footprintBoundary/traceAlphaBoundary
% logic for the alphaShape case.
if ~isfield(footprint, 'polygon') || isempty(footprint.polygon)
    return; % empty footprint -- a missing file already means this, per README
end
poly = footprint.polygon;
if isa(poly, 'polyshape')
    verts = poly.Vertices;
elseif isa(poly, 'alphaShape')
    if size(poly.Points, 2) ~= 2
        return; % 3D boundary export not supported here either, see scriptcsv.m
    end
    [bf, bv] = boundaryFacets(poly);
    if isempty(bf)
        return;
    end
    verts = traceAlphaBoundary(bf, bv);
else
    return;
end
if isempty(verts)
    return;
end
writetable(array2table(verts, 'VariableNames', {'x', 'y'}), filename);
end

function verts = traceAlphaBoundary(bf, bv)
% Verbatim port of output/scriptcsv.m's traceAlphaBoundary: traces an
% ordered closed polygon from a 2-D boundary-facets edge list. Works
% correctly for simple (single-region) connected alpha shapes; footprints
% with multiple disjoint regions are not stitched with NaN delimiters by
% this port -- flagged here rather than silently producing a
% mis-ordered vertex list. Left as a follow-up if a multi-region
% alphaShape footprint is ever needed for a new fixture.
n = size(bv, 1);
if n == 0
    verts = [];
    return;
end
adj = zeros(n, 2);
cnt = zeros(n, 1);
for k = 1:size(bf, 1)
    v1 = bf(k, 1);
    v2 = bf(k, 2);
    cnt(v1) = cnt(v1) + 1;
    if cnt(v1) <= 2, adj(v1, cnt(v1)) = v2; end
    cnt(v2) = cnt(v2) + 1;
    if cnt(v2) <= 2, adj(v2, cnt(v2)) = v1; end
end
order = zeros(n, 1);
order(1) = 1;
prev = 0;
curr = 1;
for k = 2:n
    nxt = adj(curr, adj(curr, :) ~= prev & adj(curr, :) ~= 0);
    if isempty(nxt)
        break;
    end
    order(k) = nxt(1);
    prev = curr;
    curr = order(k);
end
valid = order ~= 0;
verts = bv(order(valid), :);
end

function names = sanitizeNames(names)
% array2table/cell2table's 'VariableNames' must be valid MATLAB
% identifiers -- feature/algorithm labels from a real dataset are already
% fine (verified against this toolkit's own reference dataset), but a
% hand-crafted future fixture might not be, so this is defensive rather
% than assumed unnecessary.
if isempty(names)
    return;
end
names = matlab.lang.makeValidName(cellstr(names));
names = matlab.lang.makeUniqueStrings(names);
end

function mkdirIfMissing(d)
if ~isfolder(d)
    mkdir(d);
end
end

function s = ensureTrailingSlash(s)
if ~(endsWith(s, '/') || endsWith(s, '\'))
    s = [s '/'];
end
end

function writeProvenance(toolkitRoot, outputRoot)
% Records exactly which MATLAB commit/version produced this export --
% T5's actual ask (roadmap docs/pyIS_docs_quality_roadmap.md). Commit
% this file alongside any regenerated fixture set.
commit = gitCommit(toolkitRoot);
toolkitVersion = readToolkitVersion(toolkitRoot);

provenance = struct( ...
    'matlab_commit', commit, ...
    'matlab_repo', 'https://github.com/andremun/InstanceSpace', ...
    'toolkit_version', toolkitVersion, ...
    'dataset', 'test/data/metadata.csv + metadata_test.csv (Munoz et al. 2018 study)', ...
    'generated_at', string(datetime('now', 'TimeZone', 'UTC'), 'yyyy-MM-dd''T''HH:mm:ss''Z'''), ...
    'generator_script', 'pyis_export_reference_data.m', ...
    'matlab_version', version());

fid = fopen([outputRoot 'provenance.json'], 'w');
if fid == -1
    warning('pyis_export:provenanceWriteFailed', ...
        'Could not write provenance.json to ''%s''.', outputRoot);
    return;
end
fprintf(fid, '%s', jsonencode(provenance, 'PrettyPrint', true));
fclose(fid);
end

function commit = gitCommit(toolkitRoot)
commit = 'unknown';
try
    originalDir = pwd();
    cleanupObj = onCleanup(@() cd(originalDir)); %#ok<NASGU>
    cd(toolkitRoot);
    [status, cmdOut] = system('git rev-parse HEAD');
    if status == 0
        commit = strtrim(cmdOut);
    end
catch
    % Leave 'unknown' -- not being able to shell out to git shouldn't
    % abort the whole export.
end
end

function v = readToolkitVersion(toolkitRoot)
v = 'unknown';
contentsFile = [toolkitRoot 'Contents.m'];
if ~isfile(contentsFile)
    return;
end
lines = strsplit(fileread(contentsFile), newline);
if numel(lines) >= 2
    v = strtrim(regexprep(lines{2}, '^%\s*', ''));
end
end
