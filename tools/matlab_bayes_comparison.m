% SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
% Copyright (c) 2026 Mario Andres Munoz Acosta and contributors
function pyis_bayes_comparison(inputPath, outputDirectory, mode)
% Controlled maintainer experiment. Exact pinned PYTHIA local helpers are
% appended by compare_bayes.py; no production MATLAB file is modified.
record = jsondecode(fileread(inputPath));
X = record.x; allY = logical(record.y);
if nargin < 3, mode = 'search'; end
if strcmp(mode,'search')
    if ~isfolder(fullfile(outputDirectory,'matlab'))
        mkdir(fullfile(outputDirectory,'matlab'));
    end
    for k=1:numel(record.cases)
        item=record.cases(k);
        output=fullfile(outputDirectory,'matlab',[item.id '.json']);
        assert(~isfile(output),'Preserving existing evidence: %s',output);
        masks=item.folds(:)==(0:record.fold_count-1);
        cp=isacompat.Partition(masks);
        y=allY(:,item.algorithm_index+1); W=ones(size(y));
        opts=struct('seed',item.seed,'parallel',false,'ncores',1,...
            'verbose',false,'ispolykrnl',false,'nTuningIter',item.budget);
        rng(item.seed,'twister');
        vars=classifierBayesVars('svm');
        objective=@(tbl) bayesObjective('svm',X,y,W,cp,tbl,true,opts,item.seed);
        starts=array2table(item.initial,'VariableNames',{'p1','p2'});
        result=bayesopt(objective,vars,'MaxObjectiveEvaluations',item.budget,...
            'NumSeedPoints',record.seed_points,'InitialX',starts,...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'Verbose',0,'PlotFcn',[],'UseParallel',false);
        assert(height(result.XTrace)==item.budget);
        initialObserved=table2array(result.XTrace(1:4,:));
        rows=repmat(struct('iteration',0,'box_constraint',0,'kernel_scale',0,...
            'cv_error',0,'running_best',0),item.budget,1);
        for j=1:item.budget
            rows(j).iteration=j;
            rows(j).box_constraint=result.XTrace.p1(j);
            rows(j).kernel_scale=result.XTrace.p2(j);
            rows(j).cv_error=result.ObjectiveTrace(j);
            rows(j).running_best=min(result.ObjectiveTrace(1:j));
        end
        run=struct('case',item.id,'backend','MATLAB bayesopt, EI-plus',...
            'objective','global out-of-fold classification error',...
            'trace',rows,'version',version,'release',version('-release'),...
            'matlab_commit',record.matlab_commit,...
            'helpers_sha256',record.matlab_helpers_sha256);
        run.initial_points=initialObserved;
        writeRecord(output,run);
        fprintf('MATLAB %s: best error %.6f\n',item.id,min(result.ObjectiveTrace));
    end
else
    assert(~isfile(fullfile(outputDirectory,'cross-results.json')),...
        'Preserving existing cross-evaluation evidence.');
    requests=jsondecode(fileread(fullfile(outputDirectory,'cross-inputs.json')));
    for k=1:numel(requests)
        request=requests(k);
        item=record.cases(strcmp({record.cases.id},request.case));
        masks=item.folds(:)==(0:record.fold_count-1);
        cp=isacompat.Partition(masks); y=allY(:,item.algorithm_index+1);
        opts=struct('seed',item.seed,'parallel',false,'ncores',1,...
            'verbose',false,'ispolykrnl',false);
        rng(item.seed,'twister');
        [pred,prob]=crossValPredict('svm',X,y,ones(size(y)),cp,...
            request.box_constraint,request.kernel_scale,opts,item.seed);
        assert(all(isfinite(prob)),'Cross-evaluation fit failed.');
        mistakes=sum((pred~=y)&masks,1); sizes=sum(masks,1);
        requests(k).matlab=struct('mistakes',mistakes,'sizes',sizes,...
            'global_error',sum(mistakes)/sum(sizes),...
            'fold_mean_error',mean(mistakes./sizes));
    end
    writeRecord(fullfile(outputDirectory,'cross-results.json'),requests);
    fprintf('MATLAB: %d cross-evaluations complete.\n',numel(requests));
end
end

function writeRecord(path,value)
fid=fopen(path,'w'); assert(fid>=0); guard=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(value));
end
