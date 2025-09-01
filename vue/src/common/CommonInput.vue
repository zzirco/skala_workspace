<template>
  <div class="m-3">
    <label :for="elementId" class="form-label" :class="getSize()">{{
      label
    }}</label>
    <input
      v-model="inputEmail"
      :type="type"
      class="form-control"
      :class="getSize()"
      :id="elementId"
      :placeholder="placeholder"
      :disabled="disabled"
      :readonly="readonly"
      @keyup.enter="emit('enterKey')"
    />
  </div>
</template>
<script setup>
import { ref } from "vue";
const props = defineProps({
  label: String,
  type: String,
  placeholder: String,
  size: String,
  readonly: { type: Boolean, default: false },
  disabled: { type: Boolean, default: false },
});
const emit = defineEmits(["enterKey"]);
const elementId = crypto.randomUUID();
const inputEmail = defineModel();
function getSize() {
  if (props.size === "sm") {
    return "form-control-sm";
  } else if (props.size === "lg") {
    return "form-control-lg";
  } else {
    return "";
  }
}
</script>
