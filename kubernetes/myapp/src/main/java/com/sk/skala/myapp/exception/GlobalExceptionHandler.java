package com.sk.skala.myapp.exception;


import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice(basePackages = "com.sk.skala.http_request_base.controller")
public class GlobalExceptionHandler {

	// 전역적으로 400 Bad Request 예외 처리
	@ExceptionHandler(BadRequestException.class)
	public ResponseEntity<String> handleBadRequest(BadRequestException ex) {
		return ResponseEntity.badRequest().body("Global 400 Error: " + ex.getMessage());
	}

	// 전역적으로 500 Internal Server Error 예외 처리
	@ExceptionHandler(InternalServerException.class)
	public ResponseEntity<String> handleInternalServerError(RuntimeException ex) {
		return ResponseEntity.internalServerError().body("Global 500 Error: " + ex.getMessage());
	}
}
