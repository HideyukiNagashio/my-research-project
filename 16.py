import time
import csv
import struct
from bleak import BleakClient, BleakScanner
import asyncio

# 各ArduinoのBluetoothアドレスを設定
RIGHT_FOOT_ADDRESS = "22:D8:A1:F4:60:03"  # 右足デバイスのBluetoothアドレス
LEFT_FOOT_ADDRESS = "51:2A:09:8C:DB:46"   # 左足デバイスのBluetoothアドレス

# 保存するCSVファイル名
RIGHT_CSV_FILENAME = "yanaze_s_right_foot_data.csv"
LEFT_CSV_FILENAME = "yanaze_s_left_foot_data.csv"

# データ記録フラグ
is_recording = False

# グローバル変数で両方のデータを受信
received_data = {"Right": [], "Left": []}

# バッチ送信の設定
BATCH_SIZE = 100  # バッチごとのデータ数

# カスタムUUID
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
COMMAND_CHARACTERISTIC_UUID = "87654321-4321-6789-4321-0fedcba98765"  # コマンド用
DATA_CHARACTERISTIC_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"     # データ用

# CSVヘッダーの定義
CSV_HEADERS = {
    "Right": ["ElapsedTime"] + [f"Right_Foot_{i+1}" for i in range(8)],
    "Left": ["ElapsedTime"] + [f"Left_Foot_{i+1}" for i in range(8)]
}

def notification_handler(device_label, data):
    global is_recording, received_data
    if is_recording:
        # 受信データを21バイトごとに分割
        for i in range(0, len(data), 21):
            chunk = data[i:i + 21]

            # データ長が21バイト未満なら無視
            if len(chunk) != 21:
                print(f"{device_label}: 不正なデータ長 {len(chunk)} バイトを受信しました: {chunk}")
                continue

            try:
                # データを解析
                identification_number = chunk[0]
                elapsed_time = struct.unpack('<I', chunk[1:5])[0]  # リトルエンディアンでuint32
                analog_values = struct.unpack('<8H', chunk[5:21])  # リトルエンディアンで8つのuint16

                # データの格納
                row_data = [elapsed_time] + list(analog_values)
                received_data[device_label].append(row_data)

                # バッチサイズに達したらCSVに保存
                if len(received_data[device_label]) >= BATCH_SIZE:
                    save_to_csv(device_label)
                    received_data[device_label] = []  # データをリセット

                # デバッグ用出力
                print(f"{device_label}: ID={identification_number}, ElapsedTime={elapsed_time}, AnalogData={analog_values}")

            except struct.error as e:
                print(f"{device_label}: データ解析エラー: {e}")


# データをCSVファイルに保存
def save_to_csv(device_label):
    filename = RIGHT_CSV_FILENAME if device_label == "Right" else LEFT_CSV_FILENAME
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダーを一度だけ書き込む
        if csvfile.tell() == 0:
            writer.writerow(CSV_HEADERS[device_label])
        # データ行をバッチで書き込む
        writer.writerows(received_data[device_label])
    print(f"{device_label} のデータを {filename} に保存しました")

# Arduinoと接続してデータを記録
async def connect_and_record():
    global is_recording

    # デバイスを探索
    print("スキャン中...")
    devices = await BleakScanner.discover()
    right_client = None
    left_client = None

    for device in devices:
        if device.address.lower() == RIGHT_FOOT_ADDRESS.lower():
            right_client = BleakClient(device)
        elif device.address.lower() == LEFT_FOOT_ADDRESS.lower():
            left_client = BleakClient(device)

    if not right_client or not left_client:
        print("両方のArduinoデバイスが見つかりませんでした")
        return

    try:
        # 両方のデバイスに接続
        await right_client.connect()
        await left_client.connect()
        print("両方のデバイスに接続成功")

        # データ特性に通知ハンドラを設定
        await right_client.start_notify(DATA_CHARACTERISTIC_UUID, lambda sender, data: notification_handler("Right", data))
        await left_client.start_notify(DATA_CHARACTERISTIC_UUID, lambda sender, data: notification_handler("Left", data))

        # コマンド特性を取得（書き込み用）
        right_command_char = right_client.services.get_service(SERVICE_UUID).get_characteristic(COMMAND_CHARACTERISTIC_UUID)
        left_command_char = left_client.services.get_service(SERVICE_UUID).get_characteristic(COMMAND_CHARACTERISTIC_UUID)

        print("Sを入力して記録開始、Eを入力して記録終了")

        while True:
            command = input("コマンドを入力 (S:開始, E:終了): ").strip().upper()
            if command == "S":
                # ファイルを初期化
                for filename, headers in [(RIGHT_CSV_FILENAME, CSV_HEADERS["Right"]), (LEFT_CSV_FILENAME, CSV_HEADERS["Left"])]:
                    with open(filename, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(headers)  # ヘッダーを書き込む
                        print("CSVファイルを初期化しました")
                        
                print("Sコマンドを送信します")  # 送信確認メッセージ
                await right_client.write_gatt_char(COMMAND_CHARACTERISTIC_UUID, b'S')
                await left_client.write_gatt_char(COMMAND_CHARACTERISTIC_UUID, b'S')
                is_recording = True
                print("記録を開始しました")
            elif command == "E":
                print("Eコマンドを送信します")  # 送信確認メッセージ
                await right_client.write_gatt_char(COMMAND_CHARACTERISTIC_UUID, b'E')
                await left_client.write_gatt_char(COMMAND_CHARACTERISTIC_UUID, b'E')
                is_recording = False
                print("記録を終了しました")
                break

        # 残りのデータを保存
        if received_data["Right"]:
            save_to_csv("Right")
        if received_data["Left"]:
            save_to_csv("Left")

        # 通知を停止
        await right_client.stop_notify(DATA_CHARACTERISTIC_UUID)
        await left_client.stop_notify(DATA_CHARACTERISTIC_UUID)

    finally:
        # デバイスの切断
        await right_client.disconnect()
        await left_client.disconnect()
        print("デバイスを切断しました")

# メイン関数の呼び出し
if __name__ == "__main__":
    asyncio.run(connect_and_record())
