<?php

namespace App\Controllers\Api;

use CodeIgniter\RESTful\ResourceController;
use CodeIgniter\HTTP\ResponseInterface;

abstract class BaseApiController extends ResourceController
{
    protected string $format = 'json';

    protected function ok(mixed $data = null, string $message = 'OK'): ResponseInterface
    {
        return $this->response->setStatusCode(200)->setJSON([
            'success' => true,
            'message' => $message,
            'data'    => $data,
        ]);
    }

    protected function created(mixed $data = null, string $message = 'Created'): ResponseInterface
    {
        return $this->response->setStatusCode(201)->setJSON([
            'success' => true,
            'message' => $message,
            'data'    => $data,
        ]);
    }

    protected function fail422(string $message, array $errors = []): ResponseInterface
    {
        return $this->response->setStatusCode(422)->setJSON([
            'success' => false,
            'message' => $message,
            'errors'  => $errors,
        ]);
    }

    protected function fail404(string $message = 'Not found'): ResponseInterface
    {
        return $this->response->setStatusCode(404)->setJSON([
            'success' => false,
            'message' => $message,
        ]);
    }

    protected function fail500(string $message, string $detail = ''): ResponseInterface
    {
        return $this->response->setStatusCode(500)->setJSON([
            'success' => false,
            'message' => $message,
            'detail'  => $detail,
        ]);
    }

    protected function jsonBody(): array
    {
        $body = $this->request->getBody();
        return json_decode($body ?: '{}', true) ?? [];
    }
}