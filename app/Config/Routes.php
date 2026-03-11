<?php

use CodeIgniter\Router\RouteCollection;

/** @var RouteCollection $routes */

$routes->get('/', 'DashboardController::index');
$routes->get('dashboard', 'DashboardController::index');

$routes->group('api/datasets', ['filter' => 'cors'], function ($routes) {
    $routes->get('/', 'Api\DatasetController::index');
    $routes->post('upload', 'Api\DatasetController::upload');
    $routes->get('(:num)', 'Api\DatasetController::show/$1');
    $routes->delete('(:num)', 'Api\DatasetController::delete/$1');
    $routes->get('(:num)/profile', 'Api\DatasetController::profile/$1');
    $routes->get('(:num)/preview', 'Api\DatasetController::preview/$1');
    $routes->get('(:num)/stats', 'Api\DatasetController::stats/$1');
    $routes->get('(:num)/correlation', 'Api\DatasetController::correlation/$1');
    $routes->get('(:num)/distribution', 'Api\DatasetController::distribution/$1');
});

$routes->group('api/engineering', ['filter' => 'cors'], function ($routes) {
    $routes->post('transform', 'Api\EngineeringController::transform');
    $routes->post('split', 'Api\EngineeringController::split');
    $routes->post('select-features', 'Api\EngineeringController::selectFeatures');
    $routes->get('transformers', 'Api\EngineeringController::listTransformers');
    $routes->post('save-pipeline', 'Api\EngineeringController::savePipeline');
    $routes->get('pipeline/(:num)', 'Api\EngineeringController::getPipeline/$1');
});

$routes->group('api/experiments', ['filter' => 'cors'], function ($routes) {
    $routes->get('/', 'Api\ExperimentController::index');
    $routes->post('/', 'Api\ExperimentController::create');
    $routes->get('(:num)', 'Api\ExperimentController::show/$1');
    $routes->delete('(:num)', 'Api\ExperimentController::delete/$1');
    $routes->post('(:num)/run', 'Api\ExperimentController::run/$1');
    $routes->get('(:num)/status', 'Api\ExperimentController::status/$1');
    $routes->get('(:num)/results', 'Api\ExperimentController::results/$1');
    $routes->get('(:num)/logs', 'Api\ExperimentController::logs/$1');
});

$routes->group('api/algorithms', ['filter' => 'cors'], function ($routes) {
    $routes->get('/', 'Api\AlgorithmController::index');
    $routes->get('(:segment)', 'Api\AlgorithmController::show/$1');
    $routes->get('(:segment)/params', 'Api\AlgorithmController::params/$1');
});

$routes->group('api/tuning', ['filter' => 'cors'], function ($routes) {
    $routes->post('start', 'Api\TuningController::start');
    $routes->get('(:num)/status', 'Api\TuningController::status/$1');
    $routes->get('(:num)/progress', 'Api\TuningController::progress/$1');
    $routes->post('(:num)/stop', 'Api\TuningController::stop/$1');
    $routes->get('(:num)/best', 'Api\TuningController::best/$1');
});

$routes->group('api/models', ['filter' => 'cors'], function ($routes) {
    $routes->get('/', 'Api\ModelController::index');
    $routes->post('train', 'Api\ModelController::train');
    $routes->get('compare', 'Api\ModelController::compare');
    $routes->get('(:num)', 'Api\ModelController::show/$1');
    $routes->delete('(:num)', 'Api\ModelController::delete/$1');
    $routes->get('(:num)/metrics', 'Api\ModelController::metrics/$1');
    $routes->get('(:num)/confusion', 'Api\ModelController::confusionMatrix/$1');
    $routes->get('(:num)/importance', 'Api\ModelController::featureImportance/$1');
    $routes->get('(:num)/roc', 'Api\ModelController::roc/$1');
    $routes->get('(:num)/learning', 'Api\ModelController::learningCurve/$1');
    $routes->post('(:num)/predict', 'Api\ModelController::predict/$1');
    $routes->post('(:num)/export', 'Api\ModelController::export/$1');
});

$routes->group('api/predict', ['filter' => 'cors'], function ($routes) {
    $routes->post('single', 'Api\PredictionController::single');
    $routes->post('batch', 'Api\PredictionController::batch');
    $routes->get('history', 'Api\PredictionController::history');
});

$routes->get('api/health', 'Api\SystemController::health');
$routes->get('api/system/info', 'Api\SystemController::info');