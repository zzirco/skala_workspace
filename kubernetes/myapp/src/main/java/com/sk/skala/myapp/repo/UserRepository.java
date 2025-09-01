package com.sk.skala.myapp.repo;

import com.sk.skala.myapp.domain.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

	List<User> findByNameIgnoreCase(String name);

	List<User> findByRegionId(Long regionId);

	List<User> findByRegionName(String regionName);

	boolean existsByEmail(String email);

}
