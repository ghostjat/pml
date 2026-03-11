<?php

namespace App\Controllers\Api;

use App\Models\ProjectModel;
use App\Models\ModelRunModel;
use App\Models\DatasetModel;
use App\Libraries\MLService;

class TrainingController extends BaseApiController
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

    // POST /api/v1/projects/:id/train
    public function train($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404('Project not found');

        $body = $this->jsonBody();

        $algorithms = $body['algorithms'] ?? [];
        if (empty($algorithms)) {
            return $this->fail422('Select at least one algorithm');
        }

        $engineerConfig = json_decode($project['engineer_config'] ?? '{}', true) ?? [];
        if (!empty($body['engineer_config'])) {
            $engineerConfig = array_merge($engineerConfig, $body['engineer_config']);
        }

        // Load dataset
        $dataset = $this->datasetModel->getDecoded((int)$project['dataset_id']);
        if (!$dataset) return $this->fail404('Dataset not found');

        $parsed      = $this->mlSvc->parseCSV($dataset['file_path']);
        $targetCol   = $project['target_column'];
        $featureCols = array_filter($parsed['headers'], fn($h) => $h !== $targetCol);
        $featureCols = array_values($featureCols);

        // Override feature selection from body
        if (!empty($body['features'])) {
            $featureCols = $body['features'];
        }

        $runIds     = [];
        $bestScore  = null;
        $bestRunId  = null;
        $taskType   = $project['task_type'];
        $cvFolds    = (int)($body['cv_folds'] ?? 5);

        // Update project status
        $this->projectModel->update((int)$projectId, ['status' => 'training']);

        foreach ($algorithms as $algoConfig) {
            $algoKey = $algoConfig['key'] ?? '';
            $params  = $algoConfig['params'] ?? [];

            $registry = MLService::algorithmRegistry();
            if (!isset($registry[$algoKey])) {
                continue;
            }

            // Insert pending run
            $runId = $this->runModel->insert([
                'project_id'  => (int)$projectId,
                'algorithm'   => $registry[$algoKey]['name'],
                'algo_class'  => $registry[$algoKey]['class'],
                'hyperparams' => json_encode($params),
                'status'      => 'training',
            ]);

            try {
                // Build RubixML labeled dataset
                $rubixDataset = $this->mlSvc->buildDataset(
                    $parsed['records'],
                    $featureCols,
                    $targetCol,
                    $parsed['types']
                );

                // Train & evaluate
                $result = $this->mlSvc->trainAndEvaluate(
                    $algoKey,
                    $params,
                    $engineerConfig,
                    $rubixDataset,
                    $cvFolds,
                    $taskType
                );

                // Save model to disk
                $modelDir  = WRITEPATH . 'models/project_' . $projectId . '/';
                if (!is_dir($modelDir)) mkdir($modelDir, 0755, true);
                $modelPath = $modelDir . $algoKey . '_' . $runId . '.rbx';
                $this->mlSvc->saveModel($result['pipeline'], $modelPath);

                // Update run
                $this->runModel->update($runId, [
                    'metrics'           => json_encode($result['metrics']),
                    'cv_scores'         => json_encode($result['cv_scores']),
                    'confusion_matrix'  => json_encode($result['confusion_matrix']),
                    'feature_importances'=> json_encode($result['feature_importances']),
                    'model_path'        => $modelPath,
                    'train_time'        => $result['train_time'],
                    'memory_peak'       => $result['memory_peak'],
                    'best_params'       => json_encode($params),
                    'status'            => 'completed',
                ]);

                $runIds[] = $runId;

                // Track best
                $score = $taskType === 'regression'
                    ? -($result['metrics']['rmse'] ?? 0)
                    : ($result['metrics']['accuracy'] ?? 0);

                if ($bestScore === null || $score > $bestScore) {
                    $bestScore = $score;
                    $bestRunId = $runId;
                }

            } catch (\Throwable $e) {
                $this->runModel->update($runId, [
                    'status'    => 'failed',
                    'error_msg' => $e->getMessage(),
                ]);
            }
        }

        // Update project best model
        $this->projectModel->update((int)$projectId, [
            'status'       => 'trained',
            'best_model_id'=> $bestRunId,
        ]);

        // Return all results
        $runs = [];
        foreach ($runIds as $rid) {
            $runs[] = $this->runModel->getDecoded($rid);
        }

        return $this->ok([
            'runs'        => $runs,
            'best_run_id' => $bestRunId,
        ], 'Training complete');
    }

    // GET /api/v1/projects/:id/status
    public function status($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404();
        return $this->ok(['status' => $project['status']]);
    }

    // GET /api/v1/projects/:id/results
    public function results($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404();
        $runs = $this->runModel->getByProject((int)$projectId);
        return $this->ok([
            'project' => $project,
            'runs'    => $runs,
        ]);
    }
}