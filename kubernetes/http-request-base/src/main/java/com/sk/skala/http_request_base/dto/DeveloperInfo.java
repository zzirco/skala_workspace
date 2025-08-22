package com.sk.skala.http_request_base.dto;

// Java 17 record 기반, 계층 구조를 Properties와 동일하게
public record DeveloperInfo(Owner owner, Team team) {

	public static record Owner(
		String name,
		String role,
		String level
	) {}

	public static record Team(
		String position,
		String detail
	) {}
}
