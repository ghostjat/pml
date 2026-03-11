<?php

namespace App\Controllers\Api;

use App\Models\ProjectModel;
use App\Models\DatasetModel;
use App\Libraries\MLService;

class ProjectController extends BaseApiController
{
    private ProjectModel $projectModel;
    private DatasetModel $datasetModel;

    public function __construct()
    {
        $this->projectModel = new ProjectModel();
        $this->datasetModel = new DatasetModel();
    }

    // POST /api/v1/projects
    public function create()
    {
        $body = $this->jsonBody();

        $rules = [
            'name'          => 'required|min_length[1]|max_length[255]',
            'dataset_id'    => 'required|integer',
            'target_column' => 'required',
            'task_type'     => 'required|in_list[classification,regression,clustering]',
        ];

        if (!$this->validate($rules, $body)) {
            return $this->fail422('Validation failed', $this->validator->getErrors());
        }

        $dataset = $this->datasetModel->find((int)$body['dataset_id']);
        if (!$dataset) {
            return $this->fail404('Dataset not found');
        }

        $id = $this->projectModel->insert([
            'name'          => $body['name'],
            'description'   => $body['description'] ?? '',
            'dataset_id'    => (int)$body['dataset_id'],
            'task_type'     => $body['task_type'],
            'target_column' => $body['target_column'],
            'status'        => 'created',
        ]);

        return $this->created($this->projectModel->find($id), 'Project created');
    }

    // GET /api/v1/projects
    public function index()
    {
        $projects = $this->projectModel->orderBy('id', 'DESC')->findAll();
        return $this->ok($projects);
    }

    // GET /api/v1/projects/:id
    public function show($id = null)
    {
        $project = $this->projectModel->getWithRuns((int)$id);
        if (!$project) return $this->fail404('Project not found');
        return $this->ok($project);
    }

    // DELETE /api/v1/projects/:id
    public function delete($id = null)
    {
        $project = $this->projectModel->find((int)$id);
        if (!$project) return $this->fail404('Project not found');
        $this->projectModel->delete((int)$id);
        return $this->ok(null, 'Project deleted');
    }
}