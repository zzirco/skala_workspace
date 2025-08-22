package com.sk.skala.http_request_base.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import lombok.Data;

@ConfigurationProperties(prefix = "developer")
@Component
@Data
public class DeveloperProperties {

    private Owner owner = new Owner();
    private Team team = new Team();

    @Data
    public static class Owner {
        private String name;
        private String role;
        private String level;
    }

    @Data
    public static class Team {
        private String position;
        private String detail;
    }
}

