<?php

namespace App\Controllers;

use App\Models\ProjectModel;
use App\Models\DatasetModel;
use App\Services\MLService;

class DashboardController extends \CodeIgniter\Controller
{
    public function index()
    {
        return view('pages/studio');
    }

    public function project(int $id)
    {
        return view('pages/studio', ['project_id' => $id]);
    }
}