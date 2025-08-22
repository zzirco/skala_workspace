package com.sk.skala.http_request_base.controller;

import java.util.List;
import java.util.Optional;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.sk.skala.http_request_base.domain.User;
import com.sk.skala.http_request_base.service.UserService;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/api")
public class UserController {

	private final UserService userService;

	public UserController(UserService userService) {
		this.userService = userService;
	}

	// 모든 사용자 조회 및 특정 사용자 이름으로 필터링
	@GetMapping("/users")
	public List<User> getAllUsers(@RequestParam Optional<String> name) {
		return userService.findAll(name);
	}

	// GET: 특정 사용자 가져오기
	@GetMapping("/users/{id}")
	public ResponseEntity<User> getUserById(@PathVariable long id) {
		return userService.findById(id)
				.map(ResponseEntity::ok)
				.orElse(ResponseEntity.notFound().build());
	}

	// POST: 사용자 추가
	@PostMapping("/users")
	public ResponseEntity<User> createUser(@Valid @RequestBody User user) {
		return new ResponseEntity<>(userService.create(user), HttpStatus.CREATED);
	}

	// DELETE: 사용자 삭제
	@DeleteMapping("/users/{id}")
	public ResponseEntity<Void> deleteUser(@PathVariable long id) {
		return userService.delete(id) ? 
			new ResponseEntity<>(HttpStatus.NO_CONTENT) : 
			new ResponseEntity<>(HttpStatus.NOT_FOUND);
	}

		/**
	 * Valid 문제 발생 시 처리하는 Exception Handler 정의
	 */
	@ExceptionHandler(MethodArgumentNotValidException.class)
	public ResponseEntity<String> handleMethodArgumentNotValid(MethodArgumentNotValidException ex) {
		System.out.println("[MethodArgumentNotValidException ExceptionalHandler] Validation Error: ");
		

		return ResponseEntity.status(500).body("sk000" + ex.getMessage());
	}

	/**
	 * 그 외 모든 예외를 처리하는 Exception Handler 정의
	 */
	@ExceptionHandler(RuntimeException.class)
	public ResponseEntity<String> handleRuntimeException(RuntimeException ex) {
		System.out.println("[RunTimeException ExceptionalHandler] Runtime Error: ");

		return ResponseEntity.status(500).body(ex.getMessage());
	}
	
	// // PUT: 사용자 정보 수정
	// @PutMapping("/users/{id}")
	// public ResponseEntity<User> updateUser(@PathVariable Long id, @RequestBody User updatedUser) {
	//     // 기존 사용자 찾기
	//     Optional<User> existingUser = users.stream()
	//         .filter(user -> user.getId() == id)
	//         .findFirst();
		
	//     if (existingUser.isPresent()) {
	//         User user = existingUser.get();
	//         // 새로운 정보로 업데이트 (id는 유지)
	//         user.setName(updatedUser.getName());
	//         user.setEmail(updatedUser.getEmail());
	//         // 필요한 다른 필드들도 업데이트
			
	//         return new ResponseEntity<>(user, HttpStatus.OK);
	//     } else {
	//         return new ResponseEntity<>(HttpStatus.NOT_FOUND);
	//     }
	// }        
		

}