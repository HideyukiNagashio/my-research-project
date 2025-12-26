import csv
import struct
from bleak import BleakClient, BleakScanner
import asyncio
from pynput import keyboard

# 各ArduinoのBluetoothアドレスを設定
RIGHT_FOOT_ADDRESS = "00C1C940-F3E6-8B97-3909-79DCCAD5EA86"  # 右足デバイスのBluetoothアドレス
LEFT_FOOT_ADDRESS = "F49B4121-FEE1-89BA-3643-A0563792CDCC"   # 左足デバイスのBluetoothアドレス

# 保存するCSVファイル名
RIGHT_CSV_FILENAME = "iwasaki_l_right_foot_data.csv"
LEFT_CSV_FILENAME = "iwasaki_l_left_foot_data.csv"

# データ記録フラグ
is_recording = False

# グローバル変数で両方のデータを受信
received_data = {"Right": [], "Left": []}


# マーカー用の変数
marker_counter = 0
pending_marker = {"Right": None, "Left": None}  # 次のデータに記録するマーカー

# バッチ送信の設定
BATCH_SIZE = 100  # バッチごとのデータ数

# カスタムUUID
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
COMMAND_CHARACTERISTIC_UUID = "87654321-4321-6789-4321-0fedcba98765"  # コマンド用
DATA_CHARACTERISTIC_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"     # データ用

# CSVヘッダーの定義
CSV_HEADERS = {
    "Right": ["ElapsedTime"] + [f"Right_kPa_{i+1}" for i in range(8)] + 
             ["Right_Accel_X", "Right_Accel_Y", "Right_Accel_Z", 
              "Right_Gyro_X", "Right_Gyro_Y", "Right_Gyro_Z", "Marker"],
    "Left": ["ElapsedTime"] + [f"Left_kPa_{i+1}" for i in range(8)] +
            ["Left_Accel_X", "Left_Accel_Y", "Left_Accel_Z", 
             "Left_Gyro_X", "Left_Gyro_Y", "Left_Gyro_Z", "Marker"]
}

# 圧力センサーの物理量変換定数
R = 180  # Ω
D = 9.5 / 1000  # mm → m
SENSOR_AREA = 3.14159265359 * (D / 2) ** 2  # m²

PRESSURE_COEFFICIENTS = {
    "Right": [0.22497882, 4.219779047, 4.309939138, 4.593853313, 
              4.248982725, 4.973555078, 3.462631472, 4.772374839],
    "Left": [3.966326901, 4.744929171, 4.15168471, 4.56279606,
             4.079466017, 3.484332969, 4.341571247, 5.130518067]
}

def on_press(key):
    """スペースキーが押されたときにマーカーを予約"""
    global marker_counter, pending_marker, is_recording
    try:
        if key == keyboard.Key.space and is_recording:
            marker_counter += 1
            pending_marker["Right"] = marker_counter
            pending_marker["Left"] = marker_counter
            print(f"\nマーカー {marker_counter} を予約しました")
    except AttributeError:
        pass

def notification_handler(device_label, data):
    """通知ハンドラ: 受信データを解析して保存"""
    global is_recording, received_data, pending_marker
    if is_recording:
        # 受信データを37バイトごとに分割
        for i in range(0, len(data), 37):
            chunk = data[i:i + 37]

            # データ長が37バイト未満なら無視
            if len(chunk) != 37:
                print(f"{device_label}: 不正なデータ長 {len(chunk)} バイトを受信しました")
                continue

            try:
                # 経過時間（ミリ秒）
                elapsed_time = struct.unpack('<I', chunk[1:5])[0]  # リトルエンディアンでuint32

                # データ受信時に圧力センサーの値をPaに変換
                analog_values = struct.unpack('<8H', chunk[5:21])  # 圧力センサー8個（uint16）

                # ADC値を圧力（Pa）に変換
                pressure_values = []
                for j, adc_value in enumerate(analog_values):
                    # 分母が非常に小さくなるのを防ぐ
                    denominator = 1023 - adc_value
                    if denominator < 1:  # 閾値を設定（1以下なら処理しない）
                        pressure_values.append(0)  # 0を設定
                        continue

                    # 導電率の計算
                    conductance = 1000 * adc_value / ((1023 - adc_value) * R)  # mS
                    # 力の計算（N）
                    force = conductance * PRESSURE_COEFFICIENTS[device_label][j]  # N
                    # 圧力の計算（kPa）
                    pressure = force / SENSOR_AREA / 1000  # kPa
                    pressure_values.append(pressure)
                
                # 加速度データ（int16）を取得してg単位に変換
                accel_raw = struct.unpack('<3h', chunk[21:27])  # 3つのint16
                accel_x = accel_raw[0] / 4096.0  # g単位
                accel_y = accel_raw[1] / 4096.0
                accel_z = accel_raw[2] / 4096.0
                
                # 角速度データ（int16）を取得してdeg/s単位に変換
                gyro_raw = struct.unpack('<3h', chunk[27:33])  # 3つのint16
                gyro_x = gyro_raw[0] / 65.536  # deg/s単位
                gyro_y = gyro_raw[1] / 65.536
                gyro_z = gyro_raw[2] / 65.536

                # マーカーの取得と追加
                marker = ""
                if pending_marker[device_label] is not None:
                    marker = pending_marker[device_label]
                    pending_marker[device_label] = None  # マーカーをリセット

                # データの格納
                row_data = [elapsed_time] + pressure_values + [
                    accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, marker
                ]
                received_data[device_label].append(row_data)

                # バッチサイズに達したらCSVに保存
                if len(received_data[device_label]) >= BATCH_SIZE:
                    save_to_csv(device_label)
                    received_data[device_label] = []  # データをリセット

            except struct.error as e:
                print(f"{device_label}: データ解析エラー: {e}")


# データをCSVファイルに保存
def save_to_csv(device_label):
    """デバイスラベルに基づいてCSVファイルにデータを保存"""
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
    """両方のArduinoデバイスに接続してデータを記録"""
    global is_recording, marker_counter, pending_marker

    # キーボードリスナーを起動
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # デバイスを探索
    print("スキャン中...")
    devices = await BleakScanner.discover()
    right_client = None
    left_client = None

    for device in devices:
        if device.address.lower() == RIGHT_FOOT_ADDRESS.lower():
            right_client = BleakClient(device)
            print(f"右足デバイスを発見: {device.name}")
        elif device.address.lower() == LEFT_FOOT_ADDRESS.lower():
            left_client = BleakClient(device)
            print(f"左足デバイスを発見: {device.name}")

    if not right_client or not left_client:
        print("両方のArduinoデバイスが見つかりませんでした")
        listener.stop()
        return

    try:
        # 両方のデバイスに接続
        await right_client.connect()
        await right_client.get_services()
        await left_client.connect()
        await left_client.get_services()
        print("両方のデバイスに接続成功")

        # データ特性に通知ハンドラを設定
        await right_client.start_notify(DATA_CHARACTERISTIC_UUID, lambda _, data: notification_handler("Right", data))
        await left_client.start_notify(DATA_CHARACTERISTIC_UUID, lambda _, data: notification_handler("Left", data))

        print("\nSを入力して記録開始，Eを入力して記録終了")
        print("記録中にスペースキーを押すとマーカーが追加されます")

        while True:
            await asyncio.sleep(0.1)  # 非同期処理のため短い待機
            command = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("コマンドを入力 (S:開始, E:終了): ").strip().upper()
            )
            if command == "S":
                # ファイルを初期化
                for filename, headers in [(RIGHT_CSV_FILENAME, CSV_HEADERS["Right"]), (LEFT_CSV_FILENAME, CSV_HEADERS["Left"])]:
                    with open(filename, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(headers)  # ヘッダーを書き込む
                print("CSVファイルを初期化しました")

                # マーカーカウンターと予約フラグをリセット
                marker_counter = 0
                pending_marker = {"Right": None, "Left": None}
                        
                print("Sコマンドを送信します")
                await right_client.write_gatt_char(COMMAND_CHARACTERISTIC_UUID, b'S')
                await left_client.write_gatt_char(COMMAND_CHARACTERISTIC_UUID, b'S')
                is_recording = True
                print("記録を開始しました（スペースキーでマーカーを追加）")
            elif command == "E":
                print("Eコマンドを送信します")
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
        listener.stop()
        print("デバイスを切断しました")

# メイン関数の呼び出し
if __name__ == "__main__":
    asyncio.run(connect_and_record())