<?php

namespace App\Controllers\Api;

use App\Models\ProjectModel;
use App\Models\ModelRunModel;
use App\Libraries\MLService;
use Rubix\ML\Datasets\Unlabeled;

class PredictController extends BaseApiController
{
    private ProjectModel  $projectModel;
    private ModelRunModel $runModel;
    private MLService     $mlSvc;

    public function __construct()
    {
        $this->projectModel  = new ProjectModel();
        $this->runModel      = new ModelRunModel();
        $this->mlSvc         = new MLService();
    }

    // POST /api/v1/projects/:id/predict
    public function predict($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404();

        $runId = $project['best_model_id'];
        if (!$runId) return $this->fail422('No trained model found. Train a model first.');

        $run = $this->runModel->find((int)$runId);
        if (!$run || !$run['model_path'] || !file_exists($run['model_path'])) {
            return $this->fail422('Model file not found. Retrain the model.');
        }

        $body = $this->jsonBody();
        $features = $body['features'] ?? [];

        if (empty($features)) {
            return $this->fail422('features array is required');
        }

        try {
            $model   = $this->mlSvc->loadModel($run['model_path']);
            $dataset = new Unlabeled([array_values($features)]);
            $preds   = $model->predict($dataset);

            $result = ['prediction' => $preds[0]];

            // Probabilities if classifier supports it
            if (method_exists($model, 'proba')) {
                $probas = $model->proba($dataset);
                $result['probabilities'] = $probas[0] ?? [];
            }

            // Log prediction
            $db = \Config\Database::connect();
            $db->table('predictions')->insert([
                'project_id'   => (int)$projectId,
                'model_run_id' => (int)$runId,
                'input_data'   => json_encode($features),
                'prediction'   => (string)$preds[0],
                'probabilities'=> json_encode($result['probabilities'] ?? null),
                'created_at'   => date('Y-m-d H:i:s'),
            ]);

            return $this->ok($result);

        } catch (\Throwable $e) {
            return $this->fail500('Prediction failed', $e->getMessage());
        }
    }

    // POST /api/v1/projects/:id/predict/batch
    public function batch($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404();

        $runId = $project['best_model_id'];
        if (!$runId) return $this->fail422('No trained model found.');

        $run = $this->runModel->find((int)$runId);
        if (!$run || !$run['model_path'] || !file_exists($run['model_path'])) {
            return $this->fail422('Model file not found.');
        }

        $body = $this->jsonBody();
        $rows = $body['rows'] ?? [];

        if (empty($rows)) {
            return $this->fail422('rows array required');
        }

        try {
            $model   = $this->mlSvc->loadModel($run['model_path']);
            $samples = array_map(fn($r) => array_values($r), $rows);
            $dataset = new Unlabeled($samples);
            $preds   = $model->predict($dataset);

            $results = [];
            foreach ($preds as $i => $pred) {
                $results[] = ['row' => $i, 'prediction' => $pred];
            }

            return $this->ok(['predictions' => $results, 'count' => count($results)]);

        } catch (\Throwable $e) {
            return $this->fail500('Batch prediction failed', $e->getMessage());
        }
    }
}