<?php

namespace App\Controllers\Api;

use App\Models\ProjectModel;
use App\Models\ModelRunModel;
use App\Models\DatasetModel;
use App\Libraries\MLService;

class ExportController extends BaseApiController
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

    // GET /api/v1/projects/:id/export/model
    public function model($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project || !$project['best_model_id']) return $this->fail404('No trained model found');

        $run = $this->runModel->find((int)$project['best_model_id']);
        if (!$run || !file_exists($run['model_path'])) return $this->fail404('Model file not found');

        return $this->response
            ->setHeader('Content-Type', 'application/octet-stream')
            ->setHeader('Content-Disposition', 'attachment; filename="' . basename($run['model_path']) . '"')
            ->setBody(file_get_contents($run['model_path']));
    }

    // GET /api/v1/projects/:id/export/code
    public function code($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404();
        if (!$project['best_model_id']) return $this->fail422('Train a model first');

        $run = $this->runModel->getDecoded((int)$project['best_model_id']);
        $dataset = $this->datasetModel->getDecoded((int)$project['dataset_id']);

        $lang    = $this->request->getGet('lang') ?? 'php';
        $code    = $this->generateCode($lang, $project, $run, $dataset);

        return $this->ok(['language' => $lang, 'code' => $code]);
    }

    // GET /api/v1/projects/:id/export/report
    public function report($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404();

        $runs = $this->runModel->getByProject((int)$projectId);
        $dataset = $this->datasetModel->getDecoded((int)$project['dataset_id']);

        $report = [
            'project'  => $project,
            'dataset'  => ['name'=>$dataset['name'],'rows'=>$dataset['rows'],'columns'=>$dataset['columns']],
            'runs'     => $runs,
            'summary'  => $this->buildSummary($runs, $project['task_type']),
            'generated_at' => date('Y-m-d H:i:s'),
        ];

        return $this->ok($report);
    }

    private function generateCode(string $lang, array $project, array $run, array $dataset): string
    {
        $modelPath = basename($run['model_path'] ?? 'model.rbx');
        $features  = $dataset['column_names'] ?? [];
        $target    = $project['target_column'];
        $featureList = array_filter($features, fn($f) => $f !== $target);

        if ($lang === 'php') {
            return $this->generatePHP($project, $run, $modelPath, $featureList);
        }
        if ($lang === 'curl') {
            return $this->generateCurl($project, $featureList);
        }
        if ($lang === 'python') {
            return $this->generatePython($project, $featureList);
        }
        return '// Unsupported language';
    }

    private function generatePHP(array $project, array $run, string $modelPath, array $features): string
    {
        $algo = $run['algorithm'] ?? 'Unknown';
        $acc  = $run['metrics']['accuracy'] ?? 'N/A';
        $featArr = implode(', ', array_map(fn($f) => "'{$f}' => \$input['{$f}']", $features));

        return <<<PHP
<?php
/**
 * RubixML Studio — Generated Prediction Controller
 * Project: {$project['name']}
 * Algorithm: {$algo}
 * Accuracy: {$acc}
 * Generated: {$project['updated_at']}
 */

namespace App\Controllers\Api;

use CodeIgniter\RESTful\ResourceController;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Datasets\Unlabeled;

class PredictController extends ResourceController
{
    protected string \$format = 'json';

    private PersistentModel \$model;

    public function __construct()
    {
        \$this->model = PersistentModel::load(
            new Filesystem(WRITEPATH . 'models/{$modelPath}')
        );
    }

    /**
     * POST /api/predict
     * Body: { "features": { {$featArr} } }
     */
    public function predict()
    {
        \$input   = \$this->request->getJSON(true);
        \$feats   = \$input['features'] ?? [];

        if (empty(\$feats)) {
            return \$this->respond(['error' => 'features required'], 422);
        }

        \$dataset     = new Unlabeled([array_values(\$feats)]);
        \$predictions = \$this->model->predict(\$dataset);

        \$result = ['prediction' => \$predictions[0]];

        // Include probabilities if available
        if (method_exists(\$this->model, 'proba')) {
            \$probas = \$this->model->proba(\$dataset);
            \$result['probabilities'] = \$probas[0] ?? [];
        }

        return \$this->respond(\$result);
    }
}
PHP;
    }

    private function generateCurl(array $project, array $features): string
    {
        $sample = json_encode(array_combine($features, array_fill(0, count($features), 0.0)), JSON_PRETTY_PRINT);
        return <<<BASH
# RubixML Studio — cURL Example
# Project: {$project['name']}

curl -X POST http://localhost:8080/api/v1/projects/{$project['id']}/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": {$sample}
  }'
BASH;
    }

    private function generatePython(array $project, array $features): string
    {
        $sample = json_encode(array_combine($features, array_fill(0, count($features), 0.0)));
        return <<<PYTHON
"""
RubixML Studio — Python Client
Project: {$project['name']}
"""
import requests

BASE_URL = "http://localhost:8080/api/v1"

# Single prediction
response = requests.post(
    f"{BASE_URL}/projects/{$project['id']}/predict",
    json={"features": {$sample}}
)
print(response.json())

# Batch prediction
rows = [
    {$sample},
    {$sample},
]
response = requests.post(
    f"{BASE_URL}/projects/{$project['id']}/predict/batch",
    json={"rows": rows}
)
print(response.json())
PYTHON;
    }

    private function buildSummary(array $runs, string $taskType): array
    {
        if (empty($runs)) return [];
        usort($runs, function($a, $b) use ($taskType) {
            $ma = $a['metrics'] ?? [];
            $mb = $b['metrics'] ?? [];
            return $taskType === 'regression'
                ? ($ma['rmse'] ?? 999) <=> ($mb['rmse'] ?? 999)
                : ($mb['accuracy'] ?? 0) <=> ($ma['accuracy'] ?? 0);
        });
        return ['best_algorithm' => $runs[0]['algorithm'] ?? '', 'run_count' => count($runs), 'top_runs' => array_slice($runs, 0, 3)];
    }
}