package com.sk.skala.myapp.service;

import com.sk.skala.myapp.domain.User;
import com.sk.skala.myapp.domain.Region;
import com.sk.skala.myapp.repo.UserRepository;
import com.sk.skala.myapp.repo.RegionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class UserService {

	private final UserRepository userRepository;
	private final RegionRepository regionRepository;

	public List<User> findAll(Optional<String> name) {
		if (name.isPresent()) {
			return userRepository.findByNameIgnoreCase(name.get());
		}
		return userRepository.findAll();
	}

	public Optional<User> findById(Long id) {
		return userRepository.findById(id);
	}

	public List<User> findByRegionId(Long regionId) {
		return userRepository.findByRegionId(regionId);
	}

	public List<User> findByRegionName(String regionName) {
		return userRepository.findByRegionName(regionName);
	}

	@Transactional
	public User create(User user) {
		if (userRepository.existsByEmail(user.getEmail())) {
			throw new IllegalArgumentException("이미 존재하는 이메일입니다: " + user.getEmail());
		}

		if (user.getRegion() != null && user.getRegion().getId() != null) {
			Region region = regionRepository.findById(user.getRegion().getId())
							.orElseThrow(() -> new IllegalArgumentException("존재하지 않는 지역입니다: " + user.getRegion().getId()));
			user.setRegion(region);
		}

		return userRepository.save(user);
	}

	@Transactional
	public Optional<User> update(Long id, User updatedUser) {
		return userRepository.findById(id)
						.map(user -> {
							user.setName(updatedUser.getName());

							if (!user.getEmail().equals(updatedUser.getEmail())) {
								if (userRepository.existsByEmail(updatedUser.getEmail())) {
									throw new IllegalArgumentException("이미 존재하는 이메일입니다: " + updatedUser.getEmail());
								}
								user.setEmail(updatedUser.getEmail());
							}

							if (updatedUser.getRegion() != null && updatedUser.getRegion().getId() != null) {
								Region region = regionRepository.findById(updatedUser.getRegion().getId())
												.orElseThrow(() -> new IllegalArgumentException("존재하지 않는 지역입니다: " + updatedUser.getRegion().getId()));
								user.setRegion(region);
							}

							return userRepository.save(user);
						});
	}

	@Transactional
	public boolean delete(Long id) {
		if (userRepository.existsById(id)) {
			userRepository.deleteById(id);
			return true;
		}
		return false;
	}
}
