package com.sk.skala.myapp.aspect;

import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.AfterThrowing;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;
import org.aspectj.lang.JoinPoint;

@Aspect
@Component
public class LoggingAspect {

	@Pointcut("execution(* com.sk.skala.http_request_base.service.*.*(..))")
	public void serviceMethods() {}

	@Before("serviceMethods()")
	public void logBefore(JoinPoint joinPoint) {
		System.out.println("[Logging Aspect: logBefore] " + joinPoint.getSignature().getName());
	}

	@AfterReturning("serviceMethods()")
	public void logAfter(JoinPoint joinPoint) {
		System.out.println("[Logging Aspect: logAfter]" + joinPoint.getSignature().getName());
	}

	@AfterThrowing("serviceMethods()")
	public void logAfterThrowing(JoinPoint joinPoint) {
		System.out.println("[Logging Aspect: logAfterThrowing] " + joinPoint.getSignature().getName());
	}

}