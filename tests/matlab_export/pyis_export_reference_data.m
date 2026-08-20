function pyis_export_reference_data(toolkitRoot, outputRoot, varargin)
% pyis_export_reference_data  Regenerate pyInstanceSpace's MATLAB reference
% fixtures from a real run of the andremun/InstanceSpace toolkit.
%
%   pyis_export_reference_data(toolkitRoot, outputRoot)
%   pyis_export_reference_data(toolkitRoot, outputRoot, 'datasetRoot', dir)
%   pyis_export_reference_data(..., 'generatorRoot', dir, 'mode', mode)
%
%   toolkitRoot - path to a checkout of https://github.com/andremun/InstanceSpace
%                 (the directory containing InstanceSpace.m/buildIS.m).
%   outputRoot  - new destination directory. It must not already exist. The
%                 exporter builds in scratch space and publishes atomically.
%   datasetRoot - (optional) directory containing metadata.csv +
%                 metadata_test.csv. Defaults to toolkitRoot/test/data/,
%                 the toolkit's own reference dataset (Munoz et al. 2018
%                 classification study) -- the same dataset that produced
%                 pyInstanceSpace's existing tests/matlab_reference/
%                 fixtures, verified directly (identical header + row
%                 counts) rather than assumed.
%
%   generatorRoot - pyInstanceSpace checkout whose commit and exporter hash
%                   are recorded. Defaults to the checkout containing this
%                   script.
%   mode          - 'verified' (default) requires clean repositories and
%                   MATLAB R2026a; 'diagnostic' permits an older or dirty
%                   environment but cannot produce a parity oracle.
%
%   The output contains manifest.json plus shared_inputs/, build_data/, and
%   explore_data/. Every data file is hashed and described in the manifest.
%   See the sibling README.md for the trust and review workflow.

% -------------------------------------------------------------------------
% Written for the Instance Space Analysis (ISA) Toolkit
% (https://github.com/andremun/InstanceSpace), to be copied into a
% checkout of that repository and run there. Lives in pyInstanceSpace
% (the Python port) because it is that repo's test-fixture tooling, not
% because it is meant to be committed to the MATLAB repo.
% -------------------------------------------------------------------------

scriptPath = [mfilename('fullpath') '.m'];
defaultGeneratorRoot = fileparts(fileparts(fileparts(scriptPath)));
p = inputParser;
addRequired(p, 'toolkitRoot', @(x) (ischar(x) || isstring(x)) && isfolder(x));
addRequired(p, 'outputRoot', @(x) ischar(x) || isstring(x));
addParameter(p, 'datasetRoot', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'generatorRoot', defaultGeneratorRoot, ...
    @(x) (ischar(x) || isstring(x)) && isfolder(x));
addParameter(p, 'mode', 'verified', ...
    @(x) any(strcmpi(char(x), {'verified', 'diagnostic'})));
parse(p, toolkitRoot, outputRoot, varargin{:});

toolkitRoot = ensureTrailingSlash(char(p.Results.toolkitRoot));
generatorRoot = ensureTrailingSlash(char(p.Results.generatorRoot));
publishRoot = char(p.Results.outputRoot);
mode = lower(char(p.Results.mode));
datasetRoot = char(p.Results.datasetRoot);
if isempty(datasetRoot)
    datasetRoot = [toolkitRoot 'test/data/'];
end
datasetRoot = ensureTrailingSlash(datasetRoot);

requiredInputs = {'metadata.csv', 'metadata_test.csv'};
for i = 1:numel(requiredInputs)
    if ~isfile([datasetRoot requiredInputs{i}])
        error('pyis_export:missingDataset', ...
            'Required input ''%s'' was not found in ''%s''.', requiredInputs{i}, datasetRoot);
    end
end
if isfile(publishRoot) || isfolder(publishRoot)
    error('pyis_export:outputExists', ...
        'outputRoot must not exist. Export into a new path: ''%s''.', publishRoot);
end

publishParent = fileparts(publishRoot);
if isempty(publishParent)
    publishParent = pwd;
end
mkdirIfMissing(publishParent);
scratchRoot = tempname(publishParent);
workRoot = tempname(publishParent);
mkdir(scratchRoot);
mkdir(workRoot);
cleanupObj = onCleanup(@() cleanupTemporaryRoots(scratchRoot, workRoot));
outputRoot = ensureTrailingSlash(scratchRoot);
pipelineRoot = ensureTrailingSlash(workRoot);

matlabState = gitState(toolkitRoot);
generatorState = gitState(generatorRoot);
matlabRelease = ['R' version('-release')];
installed = ver;
installedToolboxes = {installed.Name};
requiredToolboxes = {'MATLAB', 'Statistics and Machine Learning Toolbox', ...
    'Optimization Toolbox', 'Global Optimization Toolbox', 'Financial Toolbox'};
missingToolboxes = setdiff(requiredToolboxes, installedToolboxes, 'stable');
if ~isempty(missingToolboxes)
    error('pyis_export:missingToolbox', ...
        'Fixture export requires these missing toolboxes: %s.', ...
        strjoin(missingToolboxes, ', '));
end
if strcmp(mode, 'verified')
    if matlabState.dirty || generatorState.dirty
        error('pyis_export:dirtySource', ...
            'Verified exports require clean MATLAB and generator repositories.');
    end
    if ~strcmp(matlabRelease, 'R2026a')
        error('pyis_export:oldMatlab', ...
            'Reference-export/v2 requires MATLAB R2026a; found %s.', matlabRelease);
    end
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

mkdirIfMissing([outputRoot 'shared_inputs/reference/']);
mkdirIfMissing([outputRoot 'build_data/']);
mkdirIfMissing([outputRoot 'explore_data/']);

copyfile([datasetRoot 'metadata.csv'], [outputRoot 'shared_inputs/reference/metadata.csv']);
copyfile([datasetRoot 'metadata_test.csv'], ...
    [outputRoot 'shared_inputs/reference/metadata_test.csv']);
copyfile([datasetRoot 'metadata.csv'], [pipelineRoot 'metadata.csv']);
copyfile([datasetRoot 'metadata_test.csv'], [pipelineRoot 'metadata_test.csv']);
[~, rawAlgolabels] = readMetadataLabels([datasetRoot 'metadata.csv']);

startTime = tic;

% =========================================================================
% Base pipeline: prelim -> sifted -> pilot -> cloister, once. Every
% downstream PYTHIA/TRACE variant below re-uses this same state instead of
% re-running these (the expensive, option-invariant) stages per variant.
% =========================================================================
fprintf('[EXPORT] Building the base pipeline from %s\n', datasetRoot);
exportOpts = struct();
exportOpts.general = struct('seed', 42, 'verbose', false, 'parallel', false);
exportOpts.outputs = struct('csv', false, 'png', false, 'fig', false, 'web', false);
obj = InstanceSpace(pipelineRoot, exportOpts);
obj = obj.build('stages', {'prelim'});
preSiftedData = obj.model.data;
exportDataSnapshot(preSiftedData, obj.model.featsel.labels, ...
    [outputRoot 'build_data/prelim/default/inputs/']);
exportPrelimArtifacts(obj.model.prelim, obj.model.featsel.labels, rawAlgolabels, ...
    obj.model.data, [outputRoot 'build_data/prelim/default/outputs/']);

obj = obj.build('stages', {'sifted'});
exportSiftedInputs(preSiftedData, [outputRoot 'build_data/sifted/default/inputs/']);
exportSiftedArtifacts(obj.model.sifted, [outputRoot 'build_data/sifted/default/outputs/']);

prePilotObj = obj; % snapshot shared by the independent PILOT evidence variants below
obj = obj.build('stages', {'pilot'});
exportPilotInputs(obj.model.data, [outputRoot 'build_data/pilot/default/inputs/']);
exportPilotArtifacts(obj.model.pilot, [outputRoot 'build_data/pilot/default/outputs/']);

obj = obj.build('stages', {'cloister'});
exportCloisterInputs(obj.model.data.X, obj.model.pilot.A, ...
    [outputRoot 'build_data/cloister/default/inputs/']);
exportCloisterArtifacts(obj.model.cloist, ...
    [outputRoot 'build_data/cloister/default/outputs/']);

% 'default/' is the only variant these four stages ever have (their output
% doesn't depend on opts.pythia/opts.trace) -- written anyway, rather than
% left stage-only, so every stage sits at the same build_data/<stage>/
% <variant>/ depth as pythia/trace below.
% =========================================================================
% Required downstream variants: current defaults, TRACE3's explicit
% PYTHIA-skip fallback, and retained legacy TRACE.
% =========================================================================
variants = { ...
    struct('name', 'trace3_default', ...
           'desc', 'MATLAB''s own untouched defaults: KNN classifier, Sobol tuning, TRACE3.', ...
           'pythia', struct(), ...
           'trace', struct('method', 'trace3', 'PI', 0.6, ...
                           'minInstances', 4, 'minAreaFrac', 0.01, 'contra', false)), ...
    struct('name', 'trace3_pythia_skip', ...
           'desc', 'TRACE3 true-label fallback with PYTHIA explicitly skipped.', ...
           'pythia', struct('skip', true), ...
           'trace', struct('method', 'trace3', 'PI', 0.6, ...
                           'minInstances', 4, 'minAreaFrac', 0.01, 'contra', false)), ...
    struct('name', 'legacy_svm', ...
           'desc', 'Retained legacy TRACE with SVM and Sobol tuning.', ...
           'pythia', struct('classifier', 'svm', 'tuning', 'sobol', ...
                            'ispolykrnl', false), ...
           'trace', struct('method', 'legacy', 'PI', 0.55, ...
                           'minInstances', 4, 'minAreaFrac', 0.01, 'contra', true)) ...
};

% Additive PILOT evidence variants for #262.  X0 and precalcAlpha are
% effective MATLAB options (PILOT.m consumes them directly), so they are
% retained in each variant's complete resolved option tree as well as
% exported as explicit stage inputs.  X0 deliberately has three columns
% while ntries is one: this proves MATLAB's documented rule that a valid
% X0 column count overrides ntries.  One restart is enough for the separate
% viewpoint optimisations and keeps fixture generation bounded.
nPilotFeatures = size(prePilotObj.model.data.X, 2);
nPilotAlgorithms = size(prePilotObj.model.data.Y, 2);
if nPilotAlgorithms < 2
    error('pyis_export:insufficientAlgorithms', ...
        'Grouped PILOT viewpoint evidence requires at least two algorithms.');
end
pilotEvidenceNtries = 1;
pilotX0Trials = 3;
pilotX0Rows = 3 * (2 * nPilotFeatures + nPilotAlgorithms);
pilotX0 = deterministicStarts(pilotX0Rows, pilotX0Trials);
groupSplit = max(1, floor(nPilotAlgorithms / 2) - 1);
pilotEvidenceVariants = { ...
    struct('name', 'pilot_standard_analytic_3d', ...
           'desc', 'Three-dimensional standard PILOT analytic solution with the default global viewpoint.', ...
           'pilot', struct('method', 'standard', 'dims', 3, 'analytic', true, ...
                           'ntries', pilotEvidenceNtries, 'viewGroups', {{}}), ...
           'solverInput', 'none'), ...
    struct('name', 'pilot_standard_numerical_3d_x0', ...
           'desc', 'Three-dimensional standard PILOT numerical solution from explicit deterministic X0.', ...
           'pilot', struct('method', 'standard', 'dims', 3, 'analytic', false, ...
                           'ntries', pilotEvidenceNtries, 'viewGroups', {{}}, 'X0', pilotX0), ...
           'solverInput', 'x0'), ...
    struct('name', 'pilot_standard_numerical_3d_precalc', ...
           'desc', 'Three-dimensional standard PILOT replay of the best exported numerical solution.', ...
           'pilot', struct('method', 'standard', 'dims', 3, 'analytic', false, ...
                           'ntries', pilotEvidenceNtries, 'viewGroups', {{}}), ...
           'solverInput', 'precalc'), ...
    struct('name', 'pilot_pls_2d', ...
           'desc', 'Two-dimensional PILOT partial least squares solution from MATLAB SIMPLS.', ...
           'pilot', struct('method', 'pls', 'dims', 2, 'analytic', false, ...
                           'ntries', pilotEvidenceNtries, 'viewGroups', {{}}), ...
           'solverInput', 'none'), ...
    struct('name', 'pilot_pls_3d_grouped', ...
           'desc', 'Three-dimensional PILOT partial least squares solution with two grouped viewpoints.', ...
           'pilot', struct('method', 'pls', 'dims', 3, 'analytic', true, 'alpha', 3.0, ...
                           'ntries', pilotEvidenceNtries, ...
                           'viewGroups', {{1:groupSplit, groupSplit+1:nPilotAlgorithms}}), ...
           'solverInput', 'none') ...
};

baseObj = obj; % snapshot with prelim/sifted/pilot/cloister already completed
resolvedVariantRecords = cell(1, numel(variants) + numel(pilotEvidenceVariants));
for v = 1:numel(variants)
    variant = variants{v};
    fprintf('[EXPORT] === PYTHIA/TRACE variant ''%s'': %s ===\n', variant.name, variant.desc);
    obj = baseObj;
    fields = fieldnames(variant.pythia);
    for f = 1:numel(fields)
        obj.opts.pythia.(fields{f}) = variant.pythia.(fields{f});
    end
    fields = fieldnames(variant.trace);
    for f = 1:numel(fields)
        obj.opts.trace.(fields{f}) = variant.trace.(fields{f});
    end
    % Re-run the toolkit's own validation/default resolution after applying
    % variant overrides. The artifact below is the complete effective tree
    % actually stored on the trained model, not a partial override list.
    obj.opts = ISAdefaults(ISAvalidateOpts(obj.opts));

    % ---- Build path (training) ----
    obj = obj.build('stages', {'pythia', 'trace'});
    resolvedPath = ['resolved_options/' variant.name '.json'];
    writeJson(struct( ...
        'schema_version', 'pyinstancespace.resolved-options/v1', ...
        'name', variant.name, ...
        'description', variant.desc, ...
        'options', obj.model.opts), ...
        [outputRoot resolvedPath]);
    resolvedVariantRecords{v} = struct( ...
        'name', variant.name, ...
        'description', variant.desc, ...
        'path', resolvedPath);
    exportPythiaInputs(obj.model, ...
        [outputRoot 'build_data/pythia/' variant.name '/inputs/']);
    exportPythiaArtifacts(obj.model.pythia, obj.model.data.algolabels, ...
        [outputRoot 'build_data/pythia/' variant.name '/outputs/']);
    exportTraceInputs(obj.model, ...
        [outputRoot 'build_data/trace/' variant.name '/inputs/']);
    exportTraceArtifacts(obj.model.trace, obj.model.data.algolabels, ...
        [outputRoot 'build_data/trace/' variant.name '/outputs/']);

    % ---- Explore path (test-set inference on the model just trained) ----
    % Same build_data/<stage>/<variant>/ shape as the build path above --
    % split by stage (pythia, trace) rather than one flat per-variant
    % folder, so build_data/ and explore_data/ are structurally identical
    % and neither name has to be remembered as the "odd one out".
    obj = obj.explore(pipelineRoot);
    testOut = obj.getResults(1);
    exportPythiaInputs(testOut, ...
        [outputRoot 'explore_data/pythia/' variant.name '/inputs/']);
    exportPythiaExploreArtifacts(testOut, ...
        [outputRoot 'explore_data/pythia/' variant.name '/outputs/']);
    exportTraceInputs(testOut, ...
        [outputRoot 'explore_data/trace/' variant.name '/inputs/']);
    exportTraceExploreArtifacts(testOut, ...
        [outputRoot 'explore_data/trace/' variant.name '/outputs/']);
end

% =========================================================================
% PILOT dimensionality/method/viewpoint evidence.  Each variant is built
% from the same post-SIFTED snapshot.  A complete downstream build is still
% required because InstanceSpace.explore intentionally rejects partial
% models; PYTHIA skip avoids unrelated classifier fitting while retaining a
% genuine public explore-path projection.
% =========================================================================
bestNumericalAlpha = [];
for v = 1:numel(pilotEvidenceVariants)
    variant = pilotEvidenceVariants{v};
    fprintf('[EXPORT] === PILOT evidence variant ''%s'': %s ===\n', ...
        variant.name, variant.desc);
    obj = prePilotObj;
    fields = fieldnames(variant.pilot);
    for f = 1:numel(fields)
        obj.opts.pilot.(fields{f}) = variant.pilot.(fields{f});
    end
    if strcmp(variant.solverInput, 'precalc')
        if isempty(bestNumericalAlpha)
            error('pyis_export:missingPrecalculatedPilot', ...
                'The X0 evidence variant must run before precalc replay.');
        end
        obj.opts.pilot.precalcAlpha = bestNumericalAlpha;
    end
    obj.opts.pythia.skip = true;
    obj.opts.trace.method = 'trace3';
    obj.opts.trace.PI = 0.6;
    obj.opts.trace.minInstances = 4;
    obj.opts.trace.minAreaFrac = 0.01;
    obj.opts.trace.contra = false;
    obj.opts = ISAdefaults(ISAvalidateOpts(obj.opts));

    isPLS = strcmpi(obj.opts.pilot.method, 'pls');
    if isPLS
        % PRELIM intentionally centres the reference study almost exactly.
        % A deterministic nonzero shift makes this stage oracle sensitive to
        % SIMPLS's mandatory internal centring instead of allowing an
        % uncentred implementation to pass accidentally.
        obj.model.data = shiftedPilotData(obj.model.data);
    end
    obj = obj.build('stages', {'pilot', 'cloister', 'pythia', 'trace'});
    pilotData = obj.model.data;
    pilotOut = obj.model.pilot;
    resolvedOptions = obj.model.opts;
    if strcmp(variant.solverInput, 'x0')
        [~, bestIdx] = max(pilotOut.perf);
        bestNumericalAlpha = pilotOut.alpha(:, bestIdx);
    end

    resolvedPath = ['resolved_options/' variant.name '.json'];
    writeJson(struct( ...
        'schema_version', 'pyinstancespace.resolved-options/v1', ...
        'name', variant.name, ...
        'description', variant.desc, ...
        'options', resolvedOptions), ...
        [outputRoot resolvedPath]);
    resolvedVariantRecords{numel(variants) + v} = struct( ...
        'name', variant.name, ...
        'description', variant.desc, ...
        'path', resolvedPath);

    buildRoot = [outputRoot 'build_data/pilot/' variant.name '/'];
    exportPilotInputs(pilotData, [buildRoot 'inputs/']);
    exportPilotSolverInputs(resolvedOptions.pilot, variant.solverInput, ...
        [buildRoot 'inputs/']);
    exportPilotStageContext(isPLS, nPilotFeatures, nPilotAlgorithms, ...
        [buildRoot 'inputs/stage_context.json']);
    exportPilotArtifacts(pilotOut, [buildRoot 'outputs/'], ...
        pilotData.algolabels);

    obj = obj.explore(pipelineRoot);
    testOut = obj.getResults(1);
    exploreRoot = [outputRoot 'explore_data/pilot/' variant.name '/'];
    exportPilotExploreInputs(testOut, obj.model, [exploreRoot 'inputs/']);
    exportPilotExploreArtifacts(testOut, [exploreRoot 'outputs/']);
end

rmdir(workRoot, 's');
writeManifest(toolkitRoot, scriptPath, outputRoot, mode, ...
    resolvedVariantRecords, matlabState, generatorState, matlabRelease, ...
    installedToolboxes, requiredToolboxes);
[moved, moveMessage] = movefile(scratchRoot, publishRoot);
if ~moved
    error('pyis_export:publishFailed', ...
        'Could not publish the completed bundle: %s', moveMessage);
end

fprintf('[EXPORT] Completed in %.1f s. Output written to %s\n', toc(startTime), publishRoot);
fprintf('EOF:SUCCESS\n');
end

% =========================================================================
% Per-stage export functions
% =========================================================================

function exportDataSnapshot(data, featlabels, destDir)
mkdirIfMissing(destDir);
writeMatrixCSV(data.Xraw, featlabels, data.instlabels(:), [destDir 'x_raw.csv']);
writeMatrixCSV(data.Yraw, data.algolabels, data.instlabels(:), [destDir 'y_raw.csv']);
writeMatrixCSV(data.X, featlabels, data.instlabels(:), [destDir 'x_processed.csv']);
writeMatrixCSV(data.Y, data.algolabels, data.instlabels(:), [destDir 'y_processed.csv']);
writeMatrixCSV(double(data.Ybin), data.algolabels, data.instlabels(:), [destDir 'y_bin.csv']);
writeMatrixCSV(data.Ybest(:), {'y_best'}, data.instlabels(:), [destDir 'y_best.csv']);
writeMatrixCSV(data.P(:), {'p_best_algo'}, data.instlabels(:), [destDir 'p.csv']);
writeMatrixCSV(double(data.beta(:)), {'beta'}, data.instlabels(:), [destDir 'beta.csv']);
writeTextCSV(featlabels, 'feature_name', [destDir 'feature_labels.csv']);
writeTextCSV(data.algolabels, 'algorithm_name', [destDir 'algorithm_labels.csv']);
end

function exportSiftedInputs(data, destDir)
mkdirIfMissing(destDir);
writeMatrixCSV(data.X, data.featlabels, data.instlabels(:), [destDir 'x.csv']);
writeMatrixCSV(data.Y, data.algolabels, data.instlabels(:), [destDir 'y.csv']);
writeMatrixCSV(double(data.Ybin), data.algolabels, data.instlabels(:), [destDir 'y_bin.csv']);
writeTextCSV(data.featlabels, 'feature_name', [destDir 'feature_labels.csv']);
end

function exportPilotInputs(data, destDir)
mkdirIfMissing(destDir);
writeMatrixCSV(data.X, data.featlabels, data.instlabels(:), [destDir 'x.csv']);
writeMatrixCSV(data.Y, data.algolabels, data.instlabels(:), [destDir 'y.csv']);
writeTextCSV(data.featlabels, 'feature_name', [destDir 'feature_labels.csv']);
end

function exportCloisterInputs(X, A, destDir)
mkdirIfMissing(destDir);
writeMatrixCSV(X, [], [], [destDir 'x.csv']);
writeMatrixCSV(A, [], [], [destDir 'projection_a.csv']);
end

function exportPythiaInputs(model, destDir)
mkdirIfMissing(destDir);
writeMatrixCSV(model.pilot.Z, coordinateLabels(size(model.pilot.Z, 2)), ...
    model.data.instlabels(:), ...
    [destDir 'z.csv']);
writeMatrixCSV(model.data.Yraw, model.data.algolabels, model.data.instlabels(:), ...
    [destDir 'y_raw.csv']);
writeMatrixCSV(double(model.data.Ybin), model.data.algolabels, model.data.instlabels(:), ...
    [destDir 'y_bin.csv']);
writeMatrixCSV(model.data.Ybest(:), {'y_best'}, model.data.instlabels(:), ...
    [destDir 'y_best.csv']);
writeTextCSV(model.data.algolabels, 'algorithm_name', [destDir 'algorithm_labels.csv']);
end

function exportTraceInputs(model, destDir)
mkdirIfMissing(destDir);
writeMatrixCSV(model.pilot.Z, coordinateLabels(size(model.pilot.Z, 2)), ...
    model.data.instlabels(:), ...
    [destDir 'z.csv']);
writeMatrixCSV(double(model.data.Ybin), model.data.algolabels, model.data.instlabels(:), ...
    [destDir 'y_bin.csv']);
writeMatrixCSV(double(model.pythia.Yhat), model.data.algolabels, model.data.instlabels(:), ...
    [destDir 'y_hat.csv']);
writeMatrixCSV(model.data.P(:), {'p_best_algo'}, model.data.instlabels(:), [destDir 'p.csv']);
writeMatrixCSV(double(model.data.beta(:)), {'beta'}, model.data.instlabels(:), ...
    [destDir 'beta.csv']);
writeTextCSV(model.data.algolabels, 'algorithm_name', [destDir 'algorithm_labels.csv']);
end

function exportPrelimArtifacts(prelimOut, featlabels, algolabels, data, destDir)
% Exports PRELIM's per-feature/per-algorithm fit parameters and its
% per-instance outputs. Field names verified directly against
% core/PRELIM.m's out.* assignments, not assumed.
mkdirIfMissing(destDir);

featTable = table( ...
    featlabels(:), prelimOut.minX(:), prelimOut.lambdaX(:), prelimOut.muX(:), ...
    prelimOut.sigmaX(:), prelimOut.medval(:), prelimOut.iqrange(:), prelimOut.hibound(:), ...
    prelimOut.lobound(:), ...
    'VariableNames', {'feature_name', 'min_x', 'lambda_x', 'mu_x', 'sigma_x', 'medval', ...
                       'iqrange', 'hi_bound', 'lo_bound'});
writetable(featTable, [destDir 'prelim_feature_params.csv']);

algoTable = table( ...
    algolabels(:), prelimOut.lambdaY(:), prelimOut.muY(:), prelimOut.sigmaY(:), ...
    'VariableNames', {'algo_name', 'lambda_y', 'mu_y', 'sigma_y'});
writetable(algoTable, [destDir 'prelim_algo_params.csv']);

writetable(table(prelimOut.minY, 'VariableNames', {'min_y'}), [destDir 'prelim_scalars.csv']);

instTable = table( ...
    data.instlabels(:), prelimOut.Ybest(:), prelimOut.P(:), prelimOut.numGoodAlgos(:), ...
    prelimOut.beta(:), ...
    'VariableNames', {'instance_id', 'y_best', 'p_best_algo', 'num_good_algos', 'beta'});
writetable(instTable, [destDir 'prelim_instance_outputs.csv']);
writeMatrixCSV(prelimOut.Ybin, algolabels, data.instlabels(:), [destDir 'prelim_ybin.csv']);
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
if isfield(siftedOut, 'clust') && ~isempty(siftedOut.clust)
    writeMatrixCSV(double(siftedOut.clust), [], [], [destDir 'sifted_clust_membership.csv']);
end
if isfield(siftedOut, 'selvars')
    writeMatrixCSV(siftedOut.selvars(:), {'selected_index'}, [], ...
        [destDir 'selected_indices.csv']);
end
end

function exportPilotArtifacts(pilotOut, destDir, varargin)
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
writeMatrixCSV(pilotOut.Z, coordinateLabels(size(pilotOut.Z, 2)), [], ...
    [destDir 'pilot_z.csv']);
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
if isfield(pilotOut, 'viewpoint') && ~isempty(pilotOut.viewpoint)
    if isempty(varargin)
        error('pyis_export:missingPilotAlgorithmLabels', ...
            'Viewpoint export requires algorithm labels.');
    end
    exportPilotViewpointArtifacts(pilotOut.viewpoint, varargin{1}, destDir);
end
end

function exportPilotSolverInputs(pilotOpts, solverInput, destDir)
mkdirIfMissing(destDir);
if strcmp(solverInput, 'x0')
    writeMatrixCSV(pilotOpts.X0, [], [], [destDir 'x0.csv']);
elseif strcmp(solverInput, 'precalc')
    writeMatrixCSV(pilotOpts.precalcAlpha, {'precalc_alpha'}, [], ...
        [destDir 'precalc_alpha.csv']);
elseif ~strcmp(solverInput, 'none')
    error('pyis_export:unknownPilotSolverInput', ...
        'Unknown PILOT solver-input mode ''%s''.', solverInput);
end
end

function exportPilotStageContext(isPLS, nfeatures, nalgorithms, filename)
if isPLS
    transform = 'deterministic-column-shift';
    featureShift = 0.25 * (1:nfeatures);
    algorithmShift = 0.4 * (1:nalgorithms);
else
    transform = 'none';
    featureShift = [];
    algorithmShift = [];
end
context = struct( ...
    'schema_version', 'pyinstancespace.pilot-evidence-context/v1', ...
    'scope', 'pilot-stage', ...
    'upstream_snapshot', 'build_data/pilot/default/inputs', ...
    'sifted_effective_pilot_dims', 2, ...
    'input_transform', transform, ...
    'feature_shift', featureShift, ...
    'algorithm_shift', algorithmShift, ...
    'explore_projection', 'InstanceSpace.explore: Z=X*A'' (uncentred)');
writeJson(context, filename);
end

function exportPilotExploreInputs(testOut, trainedModel, destDir)
mkdirIfMissing(destDir);
writeMatrixCSV(testOut.data.X, trainedModel.data.featlabels, ...
    testOut.data.instlabels(:), [destDir 'x.csv']);
writeMatrixCSV(trainedModel.pilot.A, trainedModel.data.featlabels, ...
    coordinateLabels(size(trainedModel.pilot.A, 1)), ...
    [destDir 'projection_a.csv']);
end

function exportPilotExploreArtifacts(testOut, destDir)
mkdirIfMissing(destDir);
writeMatrixCSV(testOut.pilot.Z, coordinateLabels(size(testOut.pilot.Z, 2)), ...
    testOut.data.instlabels(:), [destDir 'pilot_z.csv']);
end

function exportPilotViewpointArtifacts(viewpointOut, algolabels, destDir)
ngroups = numel(viewpointOut.groups);
groupRows = cell(0, 4);
matrixRows = zeros(2 * ngroups, 5);
for g = 1:ngroups
    members = viewpointOut.groups{g};
    for member = 1:numel(members)
        groupRows(end+1, :) = {g, member, members(member), ...
            algolabels{members(member)}}; %#ok<AGROW>
    end
    rows = (2*g-1):(2*g);
    matrixRows(rows, :) = [repmat(g, 2, 1), (1:2)', viewpointOut.A{g}];
end
groupTable = cell2table(groupRows, 'VariableNames', ...
    {'group', 'member', 'algorithm_index', 'algorithm'});
writetable(groupTable, [destDir 'viewpoint_groups.csv']);
matrixTable = array2table(matrixRows, 'VariableNames', ...
    {'group', 'view_dimension', 'z_1', 'z_2', 'z_3'});
writetable(matrixTable, [destDir 'viewpoint_a.csv']);
angleTable = table((1:ngroups)', viewpointOut.azimuth(:), viewpointOut.elevation(:), ...
    'VariableNames', {'group', 'azimuth', 'elevation'});
writetable(angleTable, [destDir 'viewpoint_angles.csv']);
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
writeMatrixCSV(pythiaOut.mu, [], [], [destDir 'normalization_mu.csv']);
writeMatrixCSV(pythiaOut.sigma, [], [], [destDir 'normalization_sigma.csv']);

metrics = table(algolabels(:), pythiaOut.accuracy(:), pythiaOut.precision(:), ...
    pythiaOut.recall(:), 'VariableNames', ...
    {'algorithm', 'accuracy', 'precision', 'recall'});
if isfield(pythiaOut, 'cvcmat') && size(pythiaOut.cvcmat, 2) == 4
    % PYTHIA stores cm(:)' in MATLAB column-major order: TN,FN,FP,TP.
    % Export named columns rather than preserving that storage order.
    metrics.true_negative = pythiaOut.cvcmat(:, 1);
    metrics.false_positive = pythiaOut.cvcmat(:, 3);
    metrics.false_negative = pythiaOut.cvcmat(:, 2);
    metrics.true_positive = pythiaOut.cvcmat(:, 4);
end
writetable(metrics, [destDir 'raw_metrics.csv']);

if isfield(pythiaOut, 'param1') && ~isempty(pythiaOut.param1)
    paramTable = table(algolabels(:), pythiaOut.param1(:), ...
        'VariableNames', {'algo', 'param1'});
    if isfield(pythiaOut, 'param2') && ~isempty(pythiaOut.param2)
        paramTable.param2 = pythiaOut.param2(:);
    end
    writetable(paramTable, [destDir 'hyperparameters.csv']);
end
end

function exportTraceArtifacts(traceOut, algolabels, destDir)
mkdirIfMissing(destDir);
if isfield(traceOut, 'summary') && ~isempty(traceOut.summary)
    writeCellCSV(traceOut.summary(2:end, 2:end), traceOut.summary(1, 2:end), ...
        traceOut.summary(2:end, 1), [destDir 'summary.csv']);
end
rows = cell(0, 14);
for i = 1:numel(algolabels)
    [goodParts, goodHoles] = writeFootprintCSV(traceOut.good{i}, ...
        [destDir 'good_' algolabels{i} '.csv']);
    rows(end+1, :) = footprintMetricRow('good', algolabels{i}, ...
        traceOut.good{i}, goodParts, goodHoles); %#ok<AGROW>
    [bestParts, bestHoles] = writeFootprintCSV(traceOut.best{i}, ...
        [destDir 'best_' algolabels{i} '.csv']);
    rows(end+1, :) = footprintMetricRow('best', algolabels{i}, ...
        traceOut.best{i}, bestParts, bestHoles); %#ok<AGROW>
end
if isfield(traceOut, 'hard') && ~isempty(traceOut.hard)
    [hardParts, hardHoles] = writeFootprintCSV(traceOut.hard, [destDir 'hard.csv']);
    rows(end+1, :) = footprintMetricRow('hard', '', traceOut.hard, ...
        hardParts, hardHoles);
end
rows(end+1, :) = footprintMetricRow('space', '', traceOut.space, 0, 0);
metricNames = {'kind', 'algorithm', 'measure', 'measure_label', 'elements', ...
    'good_elements', 'density', 'purity', 'alpha_radius', 'region_threshold', ...
    'component_count', 'geometry_part_count', 'hole_count', 'empty'};
writetable(cell2table(rows, 'VariableNames', metricNames), [destDir 'raw_metrics.csv']);
end

function exportPythiaExploreArtifacts(testOut, destDir)
% Per-variant explore-path export: PYTHIA's test-set inference output for
% whichever variant trained this testOut. Sibling of exportPythiaArtifacts
% (the *build*-path export, build_data/pythia/<variant>/) at the matching
% explore_data/pythia/<variant>/ path -- explore-mode PYTHIA runs a
% genuinely different code path internally (PYTHIAevalMode in
% core/PYTHIA.m: no hyperparameter search, just applying the
% already-trained classifiers/reconciling test-only algorithms), so its own
% summary table has fewer columns (no param1/param2/param2Label) -- kept as
% its own eval_summary.csv rather than overwriting/conflated with
% build_data/pythia/<variant>/summary.csv.
mkdirIfMissing(destDir);
writeMatrixCSV(double(testOut.pythia.Yhat), testOut.data.algolabels, testOut.data.instlabels(:), ...
    [destDir 'predictions.csv']);
writeMatrixCSV(testOut.pythia.Pr0hat, testOut.data.algolabels, testOut.data.instlabels(:), ...
    [destDir 'probabilities.csv']);
if isfield(testOut.pythia, 'summary') && ~isempty(testOut.pythia.summary)
    writeCellCSV(testOut.pythia.summary(2:end, 2:end), testOut.pythia.summary(1, 2:end), ...
        testOut.pythia.summary(2:end, 1), [destDir 'eval_summary.csv']);
end
end

function exportTraceExploreArtifacts(testOut, destDir)
% Per-variant explore-path export: TRACE's test-set footprint membership
% for whichever variant trained this testOut. Sibling of
% exportTraceArtifacts (the *build*-path export,
% build_data/trace/<variant>/) at the matching
% explore_data/trace/<variant>/ path.
mkdirIfMissing(destDir);
if isfield(testOut.trace, 'summary') && ~isempty(testOut.trace.summary)
    writeCellCSV(testOut.trace.summary(2:end, 2:end), testOut.trace.summary(1, 2:end), ...
        testOut.trace.summary(2:end, 1), [destDir 'eval_summary.csv']);
end

membershipCols = [strcat('in_good_', testOut.data.algolabels(:)'), ...
    strcat('in_best_', testOut.data.algolabels(:)')];
membership = [footprintMembership(testOut.trace.good, testOut.pilot.Z), ...
    footprintMembership(testOut.trace.best, testOut.pilot.Z)];
writeMatrixCSV(double(membership), membershipCols, testOut.data.instlabels(:), ...
    [destDir 'membership.csv']);
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

function [partCount, holeCount] = writeFootprintCSV(footprint, filename)
% Serialize every disconnected component and interior ring without false
% connecting edges. Empty footprints are explicit header-only CSV files.
cycles = {};
partIds = zeros(1, 0);
ringNames = cell(1, 0);
isHole = false(1, 0);
if isfield(footprint, 'polygon') && ~isempty(footprint.polygon)
    poly = footprint.polygon;
    if isa(poly, 'polyshape') || ...
            (isa(poly, 'alphaShape') && size(poly.Points, 2) == 2)
        [cycles, partIds, ringNames, isHole] = footprintBoundaryRings(poly);
    end
end

partCount = sum(~isHole);
holeCount = sum(isHole);
partColumn = zeros(0, 1);
ringColumn = strings(0, 1);
vertexColumn = zeros(0, 1);
holeColumn = false(0, 1);
z1 = zeros(0, 1);
z2 = zeros(0, 1);
for i = 1:numel(cycles)
    coordinates = cycles{i};
    nvertices = size(coordinates, 1);
    partColumn = [partColumn; repmat(partIds(i), nvertices, 1)]; %#ok<AGROW>
    ringColumn = [ringColumn; repmat(string(ringNames{i}), nvertices, 1)]; %#ok<AGROW>
    vertexColumn = [vertexColumn; (1:nvertices)']; %#ok<AGROW>
    holeColumn = [holeColumn; repmat(isHole(i), nvertices, 1)]; %#ok<AGROW>
    z1 = [z1; coordinates(:, 1)]; %#ok<AGROW>
    z2 = [z2; coordinates(:, 2)]; %#ok<AGROW>
end
boundaryTable = table(partColumn, ringColumn, vertexColumn, holeColumn, z1, z2, ...
    'VariableNames', {'part', 'ring', 'vertex', 'is_hole', 'z_1', 'z_2'});
mkdirIfMissing(fileparts(filename));
writetable(boundaryTable, filename);
end

function cycles = splitBoundaryCoordinates(x, y)
if iscell(x)
    cycles = cell(1, numel(x));
    for i = 1:numel(x)
        cycles{i} = removeClosingVertex([x{i}(:), y{i}(:)]);
    end
    return;
end
coordinates = [x(:), y(:)];
separators = [0; find(any(isnan(coordinates), 2)); size(coordinates, 1) + 1];
cycles = {};
for i = 1:numel(separators)-1
    segment = coordinates(separators(i)+1:separators(i+1)-1, :);
    if ~isempty(segment)
        cycles{end+1} = removeClosingVertex(segment); %#ok<AGROW>
    end
end
end

function coordinates = removeClosingVertex(coordinates)
if size(coordinates, 1) > 1 && isequal(coordinates(1, :), coordinates(end, :))
    coordinates(end, :) = [];
end
end

function [cycles, partIds, ringNames, isHole] = footprintBoundaryRings(poly)
if isa(poly, 'alphaShape')
    [triangles, points] = alphaTriangulation(poly);
    polygon = polyshape();
    for i = 1:size(triangles, 1)
        coordinates = points(triangles(i, :), :);
        polygon = union(polygon, polyshape(coordinates(:, 1), coordinates(:, 2), ...
            'Simplify', false));
    end
else
    polygon = poly;
end

parts = regions(polygon);
cycles = {};
partIds = zeros(1, 0);
ringNames = cell(1, 0);
isHole = false(1, 0);
for part = 1:numel(parts)
    [x, y] = boundary(parts(part));
    partCycles = splitBoundaryCoordinates(x, y);
    if isempty(partCycles)
        continue;
    end
    areas = cellfun(@(coordinates) abs(polyarea(coordinates(:, 1), ...
        coordinates(:, 2))), partCycles);
    [~, exterior] = max(areas);
    holeNumber = 0;
    for ring = 1:numel(partCycles)
        cycles{end+1} = partCycles{ring}; %#ok<AGROW>
        partIds(end+1) = part; %#ok<AGROW>
        isHole(end+1) = ring ~= exterior; %#ok<AGROW>
        if ring == exterior
            ringNames{end+1} = 'exterior'; %#ok<AGROW>
        else
            holeNumber = holeNumber + 1;
            ringNames{end+1} = sprintf('hole_%d', holeNumber); %#ok<AGROW>
        end
    end
end
end

function row = footprintMetricRow(kind, algorithm, footprint, parts, holes)
measure = numericField(footprint, 'measure', numericField(footprint, 'area', 0));
measureLabel = textField(footprint, 'measureLabel', 'Area');
elements = numericField(footprint, 'elements', 0);
goodElements = numericField(footprint, 'goodElements', NaN);
density = numericField(footprint, 'density', 0);
purity = numericField(footprint, 'purity', 0);
alphaRadius = NaN;
regionThreshold = NaN;
componentCount = parts;
if isfield(footprint, 'polygon') && isa(footprint.polygon, 'alphaShape')
    alphaRadius = footprint.polygon.Alpha;
    regionThreshold = footprint.polygon.RegionThreshold;
    componentCount = numRegions(footprint.polygon);
elseif isfield(footprint, 'polygon') && isa(footprint.polygon, 'polyshape')
    componentCount = numel(regions(footprint.polygon));
end
empty = ~isfield(footprint, 'polygon') || isempty(footprint.polygon);
row = {kind, algorithm, measure, measureLabel, elements, goodElements, density, ...
    purity, alphaRadius, regionThreshold, componentCount, parts, holes, empty};
end

function value = numericField(container, name, fallback)
if isfield(container, name) && isnumeric(container.(name)) && isscalar(container.(name))
    value = container.(name);
else
    value = fallback;
end
end

function value = textField(container, name, fallback)
if isfield(container, name) && (ischar(container.(name)) || isstring(container.(name)))
    value = char(container.(name));
else
    value = fallback;
end
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

function labels = coordinateLabels(dims)
labels = arrayfun(@(index) sprintf('z_%d', index), 1:dims, ...
    'UniformOutput', false);
end

function X0 = deterministicStarts(rows, ntries)
state = rng;
cleanupObj = onCleanup(@() rng(state));
rng('default');
X0 = 2 * rand(rows, ntries) - 1;
end

function data = shiftedPilotData(data)
featureShift = 0.25 * (1:size(data.X, 2));
algorithmShift = 0.4 * (1:size(data.Y, 2));
data.X = data.X + featureShift;
data.Y = data.Y + algorithmShift;
end

function [featlabels, algolabels] = readMetadataLabels(filename)
metadata = readtable(filename, 'VariableNamingRule', 'preserve');
names = metadata.Properties.VariableNames;
featlabels = names(startsWith(names, 'feature_', 'IgnoreCase', true));
algolabels = names(startsWith(names, 'algo_', 'IgnoreCase', true));
featlabels = regexprep(featlabels, '^feature_', '', 'ignorecase');
algolabels = regexprep(algolabels, '^algo_', '', 'ignorecase');
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

function writeTextCSV(values, columnName, filename)
mkdirIfMissing(fileparts(filename));
textTable = table(string(values(:)), 'VariableNames', {columnName});
writetable(textTable, filename);
end

function writeJson(value, filename)
mkdirIfMissing(fileparts(filename));
encoded = jsonencode(value, 'PrettyPrint', true);
fid = fopen(filename, 'w');
if fid == -1
    error('pyis_export:jsonWriteFailed', 'Could not open ''%s'' for writing.', filename);
end
written = fprintf(fid, '%s\n', encoded);
closeStatus = fclose(fid);
if written <= 0 || closeStatus ~= 0
    error('pyis_export:jsonWriteFailed', 'Could not write complete JSON to ''%s''.', filename);
end
end

function writeManifest(toolkitRoot, scriptPath, outputRoot, mode, ...
        resolvedVariantRecords, matlabState, generatorState, matlabRelease, ...
        installedToolboxes, requiredToolboxes)
listing = dir(fullfile(outputRoot, '**', '*'));
listing = listing(~[listing.isdir]);
files = repmat(struct('path', '', 'sha256', '', 'size_bytes', 0, ...
    'media_type', '', 'role', '', 'phase', '', 'stage', '', 'variant', '', ...
    'empty', false, 'rows', 0, 'columns', 0), 1, numel(listing));
for i = 1:numel(listing)
    fullPath = fullfile(listing(i).folder, listing(i).name);
    relativePath = strrep(fullPath(numel(outputRoot)+1:end), '\', '/');
    [phase, stage, variant, role] = describeManifestPath(relativePath);
    [~, ~, extension] = fileparts(fullPath);
    if strcmpi(extension, '.csv')
        mediaType = 'text/csv';
        % The fixture format is always comma-delimited.  Letting readtable
        % infer a delimiter can misclassify underscores in one-column label
        % files and record an incorrect manifest shape.
        csvTable = readtable(fullPath, 'Delimiter', ',', ...
            'VariableNamingRule', 'preserve', 'TextType', 'string');
        rows = height(csvTable);
        columns = width(csvTable);
        empty = rows == 0;
    elseif strcmpi(extension, '.json')
        mediaType = 'application/json';
        rows = 0;
        columns = 0;
        empty = false;
    else
        error('pyis_export:unknownFileType', ...
            'Manifest cannot classify exported file ''%s''.', relativePath);
    end
    files(i) = struct('path', relativePath, 'sha256', sha256File(fullPath), ...
        'size_bytes', listing(i).bytes, 'media_type', mediaType, 'role', role, ...
        'phase', phase, 'stage', stage, 'variant', variant, 'empty', empty, ...
        'rows', rows, 'columns', columns);
end
[~, order] = sort({files.path});
files = files(order);

if strcmp(mode, 'verified')
    trust = 'matlab-verified';
else
    trust = 'matlab-diagnostic';
end
resolvedOptions = struct();
resolvedOptions.schema_version = 'pyinstancespace.resolved-options-index/v1';
resolvedOptions.variants = [resolvedVariantRecords{:}];
manifest = struct();
manifest.schema_version = 'pyinstancespace.matlab-fixtures/v1';
manifest.profile = 'pyinstancespace.reference-export/v2';
manifest.bundle_id = 'reference-current';
manifest.trust = trust;
manifest.generated_at = string(datetime('now', 'TimeZone', 'UTC'), ...
    'yyyy-MM-dd''T''HH:mm:ss.SSSXXX');
manifest.dataset = struct('name', 'InstanceSpace reference study', 'seed', 42, ...
    'training_input', 'shared_inputs/reference/metadata.csv', ...
    'test_input', 'shared_inputs/reference/metadata_test.csv');
manifest.resolved_options = resolvedOptions;
manifest.matlab = struct('repo_commit', matlabState.commit, ...
    'repo_dirty', matlabState.dirty, 'toolkit_version', readToolkitVersion(toolkitRoot), ...
    'release', matlabRelease, 'version', version(), 'platform', computer(), ...
    'installed_toolboxes', {installedToolboxes}, ...
    'required_toolboxes', {requiredToolboxes});
manifest.generator = struct('repo_commit', generatorState.commit, ...
    'repo_dirty', generatorState.dirty, ...
    'script', 'tests/matlab_export/pyis_export_reference_data.m', ...
    'script_sha256', sha256File(scriptPath));
manifest.files = files;
writeJson(manifest, [outputRoot 'manifest.json']);
end

function [phase, stage, variant, role] = describeManifestPath(relativePath)
parts = strsplit(relativePath, '/');
role = relativePath;
if strcmp(parts{1}, 'shared_inputs') || strcmp(parts{1}, 'resolved_options')
    phase = 'shared';
    stage = '';
    if numel(parts) >= 2
        variant = erase(parts{2}, '.json');
    else
        variant = 'reference';
    end
elseif strcmp(parts{1}, 'build_data') || strcmp(parts{1}, 'explore_data')
    phase = erase(parts{1}, '_data');
    if numel(parts) < 4
        error('pyis_export:badLayout', 'Unexpected exported path ''%s''.', relativePath);
    end
    stage = parts{2};
    variant = parts{3};
else
    error('pyis_export:badLayout', 'Unexpected exported path ''%s''.', relativePath);
end
end

function state = gitState(root)
originalDir = pwd();
cleanupObj = onCleanup(@() cd(originalDir));
cd(root);
[commitStatus, commitOutput] = system('git rev-parse --verify HEAD');
if commitStatus ~= 0
    error('pyis_export:unknownCommit', ...
        'Cannot resolve the Git commit for ''%s''.', root);
end
commit = lower(strtrim(commitOutput));
if isempty(regexp(commit, '^[0-9a-f]{40}$', 'once'))
    error('pyis_export:unknownCommit', ...
        'Git returned an invalid commit for ''%s'': %s', root, commit);
end
[dirtyStatus, dirtyOutput] = system('git status --porcelain --untracked-files=all');
if dirtyStatus ~= 0
    error('pyis_export:gitStatusFailed', ...
        'Cannot inspect repository cleanliness for ''%s''.', root);
end
state = struct('commit', commit, 'dirty', ~isempty(strtrim(dirtyOutput)));
end

function digest = sha256File(filename)
fid = fopen(filename, 'r');
if fid == -1
    error('pyis_export:hashReadFailed', 'Could not read ''%s'' for hashing.', filename);
end
bytes = fread(fid, Inf, '*uint8');
closeStatus = fclose(fid);
if closeStatus ~= 0
    error('pyis_export:hashReadFailed', 'Could not close ''%s'' after hashing.', filename);
end
messageDigest = java.security.MessageDigest.getInstance('SHA-256');
messageDigest.update(bytes);
rawDigest = typecast(messageDigest.digest(), 'uint8');
digest = lower(reshape(dec2hex(rawDigest, 2).', 1, []));
end

function cleanupTemporaryRoots(scratchRoot, workRoot)
if isfolder(workRoot)
    rmdir(workRoot, 's');
end
if isfolder(scratchRoot)
    rmdir(scratchRoot, 's');
end
end

function v = readToolkitVersion(toolkitRoot)
v = 'unknown';
contentsFile = [toolkitRoot 'Contents.m'];
if ~isfile(contentsFile)
    error('pyis_export:unknownToolkitVersion', ...
        'Contents.m was not found in ''%s''.', toolkitRoot);
end
lines = strsplit(fileread(contentsFile), newline);
if numel(lines) >= 2
    v = strtrim(regexprep(lines{2}, '^%\s*', ''));
end
if strcmp(v, 'unknown') || isempty(v)
    error('pyis_export:unknownToolkitVersion', ...
        'Could not read the toolkit version from ''%s''.', contentsFile);
end
end
