package com.sk.skala.http_request_base.service;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import com.sk.skala.http_request_base.domain.User;
import com.sk.skala.http_request_base.repo.FileUserRepository;
import com.sk.skala.http_request_base.repo.UserRepository;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class UserService {
  private final UserRepository userRepository;

  public UserService(FileUserRepository userRepository) {
    this.userRepository = userRepository;
  }

  public List<User> findAll(Optional<String> name) {
    Collection<User> all = userRepository.findAll();
      if (name.isPresent()) {
          String searchName = name.get();
          return all.stream()
                  .filter(user -> user.getName().equalsIgnoreCase(searchName))
                  .toList();
      }
    log.info("Finding all users");
      return new ArrayList<>(all);
  }

  public Optional<User> findById(long id) {
    log.info("Finding user by id: " + id);
    return userRepository.findById(id);
  }

  public User create(User user) {
    log.info("Creating user: " + user);
    return userRepository.create(user);
  }

  public boolean delete(Long id) {
      Optional<User> existed = userRepository.findById(id);
      userRepository.delete(id);
      log.info("Deleting user with id: " + id);
      return existed.isPresent();
  }
}
