<?php

namespace App\Controllers\Api;

use App\Models\ProjectModel;
use App\Libraries\MLService;

class EngineerController extends BaseApiController
{
    private ProjectModel $projectModel;
    private MLService    $mlSvc;

    public function __construct()
    {
        $this->projectModel = new ProjectModel();
        $this->mlSvc = new MLService();
    }

    // POST /api/v1/projects/:id/engineer
    public function apply($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404();

        $body = $this->jsonBody();

        $config = [
            'scaler'              => $body['scaler']              ?? 'z_scale',
            'imputer'             => $body['imputer']             ?? 'mean',
            'one_hot'             => $body['one_hot']             ?? true,
            'variance_threshold'  => $body['variance_threshold']  ?? 0,
            'pca_components'      => $body['pca_components']      ?? null,
            'polynomial_degree'   => $body['polynomial_degree']   ?? 1,
            'train_ratio'         => $body['train_ratio']         ?? 0.8,
            'cv_folds'            => $body['cv_folds']            ?? 5,
            'features'            => $body['features']            ?? [],
            'drop_columns'        => $body['drop_columns']        ?? [],
        ];

        $this->projectModel->update((int)$projectId, [
            'engineer_config' => json_encode($config),
        ]);

        return $this->ok($config, 'Feature engineering config saved');
    }

    // GET /api/v1/projects/:id/engineer
    public function config($projectId = null)
    {
        $project = $this->projectModel->find((int)$projectId);
        if (!$project) return $this->fail404();

        $config = json_decode($project['engineer_config'] ?? '{}', true) ?? [];
        $transformers = MLService::transformerRegistry();
        $algorithms   = MLService::algorithmRegistry();

        return $this->ok([
            'config'       => $config,
            'transformers' => $transformers,
            'algorithms'   => $algorithms,
        ]);
    }
}