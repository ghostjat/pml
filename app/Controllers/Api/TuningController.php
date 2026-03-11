<?php

namespace App\Controllers\Api;

use App\Models\ProjectModel;
use App\Models\ModelRunModel;
use App\Models\DatasetModel;
use App\Libraries\MLService;

class TuningController extends BaseApiController
{
    private ProjectModel  $projectModel;
    private ModelRunModel $runModel;
    private DatasetModel  $datasetModel;
    private MLService     $mlSvc;

    public function __construct()
    {
        $this->projectModel  = new ProjectModel();
        $this->runModel      = new ModelRunModel();
        $this->datasetModel  = new DatasetModel();
        $this->mlSvc         = new MLService();
    }

    // POST /api/v1/projects/:id/tune
    public function tune($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404('Project not found');

        $body = $this->jsonBody();
        $algoKey   = $body['algorithm'] ?? '';
        $paramGrid = $body['param_grid'] ?? [];
        $method    = $body['method'] ?? 'grid';    // grid | random
        $metric    = $body['metric'] ?? 'accuracy';
        $cvFolds   = (int)($body['cv_folds'] ?? 5);
        $maxIter   = (int)($body['max_iter'] ?? 20);

        $registry = MLService::algorithmRegistry();
        if (!isset($registry[$algoKey])) {
            return $this->fail422("Unknown algorithm: {$algoKey}");
        }
        if (empty($paramGrid)) {
            return $this->fail422('param_grid is required');
        }

        // Load dataset
        $dataset  = $this->datasetModel->getDecoded((int)$project['dataset_id']);
        $parsed   = $this->mlSvc->parseCSV($dataset['file_path']);
        $targetCol  = $project['target_column'];
        $featureCols = array_values(array_filter($parsed['headers'], fn($h) => $h !== $targetCol));
        $engineerConfig = json_decode($project['engineer_config'] ?? '{}', true) ?? [];

        $rubixDataset = $this->mlSvc->buildDataset(
            $parsed['records'],
            $featureCols,
            $targetCol,
            $parsed['types']
        );

        // Random search: subsample combinations
        if ($method === 'random') {
            $expanded = $this->cartesian($paramGrid);
            shuffle($expanded);
            $expanded = array_slice($expanded, 0, $maxIter);
            $paramGrid = ['_random' => $expanded];
        }

        try {
            $result = $this->mlSvc->gridSearch(
                $algoKey,
                $paramGrid,
                $engineerConfig,
                $rubixDataset,
                $project['task_type'],
                $metric,
                $cvFolds
            );

            // Save best params back to a model run
            $runId = $this->runModel->insert([
                'project_id' => (int)$projectId,
                'algorithm'  => $registry[$algoKey]['name'],
                'algo_class' => $registry[$algoKey]['class'],
                'hyperparams'=> json_encode($result['best']['params']),
                'best_params'=> json_encode($result['best']['params']),
                'metrics'    => json_encode($result['best']['metrics'] ?? []),
                'status'     => 'tuned',
            ]);

            return $this->ok([
                'best_params' => $result['best']['params'],
                'best_score'  => $result['best']['score'],
                'trials'      => $result['trials'],
                'run_id'      => $runId,
            ], 'Tuning complete');

        } catch (\Throwable $e) {
            return $this->fail500('Tuning failed', $e->getMessage());
        }
    }

    // GET /api/v1/projects/:id/tune
    public function results($projectId = null)
    {
        $runs = $this->runModel->where('project_id', (int)$projectId)
                               ->where('status', 'tuned')
                               ->orderBy('id', 'DESC')
                               ->findAll();
        foreach ($runs as &$r) {
            $r['best_params'] = json_decode($r['best_params'] ?? '{}', true);
            $r['metrics']     = json_decode($r['metrics'] ?? '{}', true);
        }
        return $this->ok($runs);
    }

    private function cartesian(array $grid): array
    {
        $result = [[]];
        foreach ($grid as $key => $values) {
            $tmp = [];
            foreach ($result as $existing) {
                foreach ((array)$values as $val) {
                    $tmp[] = array_merge($existing, [$key => $val]);
                }
            }
            $result = $tmp;
        }
        return $result;
    }
}