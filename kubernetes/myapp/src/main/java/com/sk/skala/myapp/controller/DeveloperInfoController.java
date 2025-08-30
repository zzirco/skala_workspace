package com.sk.skala.myapp.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sk.skala.myapp.config.DeveloperProperties;
import com.sk.skala.myapp.dto.DeveloperInfo;

@Tag(name = "DeveloperInfo API", description = "DeveloperInfo 관련 API")
@RestController
@RequestMapping("/api")
public class DeveloperInfoController {

	private final DeveloperProperties props;

	public DeveloperInfoController(DeveloperProperties props) {
		this.props = props;
	}

	@Operation(summary = "DeveloperInfo 조회", description = "DeveloperInfo를 조회합니다.")
	@GetMapping("/developer-info")
	public DeveloperInfo info() {
		var owner = props.getOwner();
		var team = props.getTeam();

		return new DeveloperInfo(
			new DeveloperInfo.Owner(owner.getName(), owner.getRole(), owner.getLevel()),
			new DeveloperInfo.Team(team.getPosition(), team.getDetail())
		);
	}
}