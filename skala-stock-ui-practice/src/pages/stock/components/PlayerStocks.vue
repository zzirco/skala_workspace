<script setup>
import { ref, reactive, onMounted } from "vue";
import apiCall from "@/scripts/api-call";
import { usePlayer } from "@/scripts/store-player";

const stockId = ref("");
const stockQuantity = ref("");
const playerMoney = ref("");
const table = reactive({
  headers: [
    { label: "주식ID", value: "stockId" },
    { label: "주식명", value: "stockName" },
    { label: "주식가격", value: "stockPrice" },
    { label: "보유수량", value: "quantity" },
  ],
  items: [],
});
const player = usePlayer();

console.log(player);

const getPlayerInfo = async () => {
  const response = await apiCall.get(
    `/api/players/${player.playerId}`,
    null,
    null
  );
  if (response.result === 0) {
    table.items = response.body.stocks;
    playerMoney.value = response.body.playerMoney;
    console.log(table.items);
  }
};

const buyPlayerStock = async () => {
  const response = await apiCall.post("/api/players/buy", null, {
    playerId: player.playerId,
    stockId: stockId.value,
    stockQuantity: stockQuantity.value,
  });
  if (response.result === 0) {
    getPlayerInfo();
    stockId.value = "";
    stockQuantity.value = "";
  }
};

const sellPlayerStock = async () => {
  const response = await apiCall.post("/api/players/sell", null, {
    playerId: player.playerId,
    stockId: stockId.value,
    stockQuantity: stockQuantity.value,
  });
  if (response.result === 0) {
    getPlayerInfo();
    stockId.value = "";
    stockQuantity.value = "";
  }
};

onMounted(() => {
  getPlayerInfo();
});
</script>

<template>
  <div class="row mt-2">
    <span class="fs-4"
      ><i class="bi bi-person m-2"></i>{{ player.playerId }} 플레이어</span
    >
  </div>
  <div class="row border-bottom">
    <div class="col d-flex justify-content-end">
      <button class="btn btn-sm btn-primary m-1" @click="getPlayerInfo">
        <i class="bi bi-arrow-counterclockwise m-2"></i>갱신
      </button>
    </div>
  </div>
  <div class="row">
    <div class="col">
      <InlineInput
        v-model="player.playerId"
        class="m-2"
        label="플레이어ID"
        :disabled="true"
      />
      <InlineInput
        v-model="playerMoney"
        class="m-2"
        label="보유금액"
        :disabled="true"
      />
    </div>
  </div>
  <div class="row g-2 align-items-center m-2 mt-0">
    <div class="col-2 d-flex justify-content-end">
      <label class="col-form-label form-control-sm p-1">보유주식목록</label>
    </div>
    <div class="col">
      <ItemsTable
        :headers="table.headers"
        :items="table.items"
        :nosetting="true"
      />
    </div>
  </div>
  <div class="row g-2 align-items-center m-2 mt-0">
    <div class="col-2 d-flex justify-content-end">
      <label class="col-form-label form-control-sm p-1">주식선택</label>
    </div>
    <div class="col">
      <InlineInput v-model="stockId" placeholder="주식ID" />
    </div>
    <div class="col">
      <InlineInput v-model="stockQuantity" placeholder="주식수량" />
    </div>
    <div class="col d-flex justify-content-start">
      <button
        class="btn btn-sm btn-outline-primary m-1"
        @click="buyPlayerStock"
        :disabled="!(stockId && stockQuantity)"
      >
        주식 구매
      </button>
      <button
        class="btn btn-sm btn-outline-primary m-1"
        @click="sellPlayerStock"
        :disabled="!(stockId && stockQuantity)"
      >
        주식 판매
      </button>
    </div>
  </div>
</template>
