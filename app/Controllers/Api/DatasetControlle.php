<?php

namespace App\Controllers\Api;

use App\Models\DatasetModel;
use App\Libraries\MLService;

class DatasetController extends BaseApiController
{
    private DatasetModel $model;
    private MLService    $mlSvc;

    public function __construct()
    {
        $this->model = new DatasetModel();
        $this->mlSvc = new MLService();
    }

    // POST /api/v1/datasets/upload
    public function upload()
    {
        $file = $this->request->getFile('file');

        if (!$file || !$file->isValid()) {
            return $this->fail422('No valid file uploaded');
        }

        $allowedTypes = ['text/csv','text/plain','application/csv','application/json'];
        if (!in_array($file->getMimeType(), $allowedTypes) && !in_array($file->getExtension(), ['csv','json'])) {
            return $this->fail422('Only CSV/JSON files are supported');
        }

        if ($file->getSizeByUnit('mb') > 100) {
            return $this->fail422('File too large (max 100MB)');
        }

        // Move file
        $newName = $file->getRandomName();
        $uploadPath = WRITEPATH . 'datasets/';
        if (!is_dir($uploadPath)) mkdir($uploadPath, 0755, true);
        $file->move($uploadPath, $newName);
        $filePath = $uploadPath . $newName;

        try {
            // Parse
            $parsed = $this->mlSvc->parseCSV($filePath);

            // Profile
            $profile = $this->mlSvc->profileDataset($parsed['records'], $parsed['headers'], $parsed['types']);

            // Detect task type from last column
            $lastCol = end($parsed['headers']);
            $lastType = $parsed['types'][$lastCol] ?? 'unknown';
            $taskType = $lastType === 'continuous' ? 'regression' : 'classification';

            $id = $this->model->insert([
                'name'          => pathinfo($file->getClientName(), PATHINFO_FILENAME),
                'original_name' => $file->getClientName(),
                'file_path'     => $filePath,
                'file_size'     => $file->getSize(),
                'mime_type'     => $file->getMimeType(),
                'rows'          => $parsed['count'],
                'columns'       => count($parsed['headers']),
                'column_names'  => json_encode($parsed['headers']),
                'column_types'  => json_encode($parsed['types']),
                'target_column' => $lastCol,
                'task_type'     => $taskType,
                'profile_json'  => json_encode($profile),
            ]);

            return $this->created([
                'id'           => $id,
                'name'         => pathinfo($file->getClientName(), PATHINFO_FILENAME),
                'rows'         => $parsed['count'],
                'columns'      => count($parsed['headers']),
                'column_names' => $parsed['headers'],
                'column_types' => $parsed['types'],
                'target_column'=> $lastCol,
                'task_type'    => $taskType,
            ], 'Dataset uploaded successfully');

        } catch (\Throwable $e) {
            @unlink($filePath);
            return $this->fail500('Failed to parse dataset', $e->getMessage());
        }
    }

    // GET /api/v1/datasets
    public function index()
    {
        $datasets = $this->model->orderBy('id', 'DESC')->findAll();
        return $this->ok($datasets);
    }

    // GET /api/v1/datasets/:id
    public function show($id = null)
    {
        $ds = $this->model->getDecoded((int)$id);
        if (!$ds) return $this->fail404('Dataset not found');
        return $this->ok($ds);
    }

    // DELETE /api/v1/datasets/:id
    public function delete($id = null)
    {
        $ds = $this->model->find((int)$id);
        if (!$ds) return $this->fail404('Dataset not found');
        if (file_exists($ds['file_path'])) @unlink($ds['file_path']);
        $this->model->delete((int)$id);
        return $this->ok(null, 'Dataset deleted');
    }

    // GET /api/v1/datasets/:id/profile
    public function profile($id = null)
    {
        $ds = $this->model->getDecoded((int)$id);
        if (!$ds) return $this->fail404('Dataset not found');
        return $this->ok($ds['profile_json']);
    }

    // GET /api/v1/datasets/:id/sample
    public function sample($id = null)
    {
        $ds = $this->model->getDecoded((int)$id);
        if (!$ds) return $this->fail404();
        $parsed = $this->mlSvc->parseCSV($ds['file_path']);
        $n = (int)($this->request->getGet('n') ?? 50);
        return $this->ok([
            'headers' => $parsed['headers'],
            'rows'    => array_slice($parsed['records'], 0, $n),
        ]);
    }

    // GET /api/v1/datasets/:id/stats
    public function stats($id = null)
    {
        $ds = $this->model->getDecoded((int)$id);
        if (!$ds) return $this->fail404();
        return $this->ok([
            'rows'          => $ds['rows'],
            'columns'       => $ds['columns'],
            'column_names'  => $ds['column_names'],
            'column_types'  => $ds['column_types'],
            'task_type'     => $ds['task_type'],
            'target_column' => $ds['target_column'],
            'missing_pct'   => $this->computeOverallMissing($ds['profile_json'] ?? []),
        ]);
    }

    // GET /api/v1/datasets/:id/corr
    public function correlation($id = null)
    {
        $ds = $this->model->getDecoded((int)$id);
        if (!$ds) return $this->fail404();
        $parsed = $this->mlSvc->parseCSV($ds['file_path']);
        $corr = $this->mlSvc->correlationMatrix($parsed['records'], $parsed['headers'], $parsed['types']);
        return $this->ok($corr);
    }

    // GET /api/v1/datasets/:id/dist?column=xxx
    public function distribution($id = null)
    {
        $ds = $this->model->getDecoded((int)$id);
        if (!$ds) return $this->fail404();
        $col = $this->request->getGet('column') ?? '';
        if (!in_array($col, $ds['column_names'])) {
            return $this->fail422("Column '{$col}' not found");
        }
        $profile = $ds['profile_json'][$col] ?? null;
        return $this->ok($profile);
    }

    private function computeOverallMissing(array $profile): float
    {
        if (empty($profile)) return 0;
        $total = array_sum(array_column($profile, 'missing_pct'));
        return round($total / count($profile), 2);
    }
}