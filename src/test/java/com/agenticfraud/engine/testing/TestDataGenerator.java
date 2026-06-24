package com.agenticfraud.engine.testing;

import com.agenticfraud.engine.models.CustomerProfile;
import com.agenticfraud.engine.models.Transaction;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.*;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;

/**
 * Test Data Generator for Streaming-Intelligent Fraud Detection Generates realistic test
 * transactions and customer profiles
 * <p>
 * Run as mvn exec:java -Dexec.mainClass="com.agenticfraud.engine.testing.TestDataGenerator" -Dexec.classpathScope="test"
 */
public class TestDataGenerator {

  private static final ObjectMapper objectMapper =
      new ObjectMapper().registerModule(new JavaTimeModule());
  private static final Random random = new Random();

  public static void main(String[] args) throws Exception {
    System.out.println("Starting Test Data Generator for Streaming-Intelligent Fraud Detection\n");

    // Setup Kafka producers
    KafkaProducer<String, String> producer = createProducer();

    // Generate customer profiles first
    System.out.println("Generating Customer Profiles...");
    List<CustomerProfile> customerProfiles = generateCustomerProfiles(1);
//    List<CustomerProfile> customerProfiles = generateCustomerProfiles(2);
    sendCustomerProfiles(producer, customerProfiles);

    System.out.println("Generating Test Transactions...");
    System.out.println("Choose scenario:");
    System.out.println("1. Normal transactions");
    System.out.println("2. High velocity attack (rapid-fire)");
    System.out.println("3. Unusual amount fraud");
    System.out.println("4. Mixed scenario");
    System.out.println("5. Continuous stream (Press Ctrl+C to stop)");

    Scanner scanner = new Scanner(System.in);
    int choice = scanner.nextInt();

    switch (choice) {
      case 1 -> generateNormalTransactions(producer, customerProfiles, 3);
      case 2 -> generateHighVelocityAttack(producer, customerProfiles.getFirst(), 15);
      case 3 -> generateUnusualAmountFraud(producer, customerProfiles.getFirst(), 5);
      case 4 -> generateMixedScenario(producer, customerProfiles);
      case 5 -> generateContinuousStream(producer, customerProfiles);
    }

    System.out.println("\nTest data generation complete!");
    producer.close();
  }

  private static KafkaProducer<String, String> createProducer() {
    Properties props = new Properties();
    props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
    props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
    props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
    props.put(ProducerConfig.ACKS_CONFIG, "all");
    return new KafkaProducer<>(props);
  }

  // ================================
  // CUSTOMER PROFILE GENERATION
  // ================================

  private static List<CustomerProfile> generateCustomerProfiles(int count) throws Exception {
    List<CustomerProfile> profiles = new ArrayList<>();

    List<String> riskLevels = List.of("HIGH");
//    List<String> riskLevels = List.of("LOW", "MEDIUM", "HIGH");
    List<String> locations = List.of("New York", "Los Angeles", "Chicago", "Houston", "Phoenix");
    List<String> categories = List.of("GROCERY", "GAS_STATION", "RESTAURANT", "RETAIL", "ONLINE");

    for (int i = 1; i <= count; i++) {
      CustomerProfile profile =
          new CustomerProfile(
              "CUST-" + String.format("%03d", i),
              BigDecimal.valueOf(50 + random.nextInt(450)), // $50-$500 average
              BigDecimal.valueOf(1000 + random.nextInt(4000)), // $1000-$5000 daily limit
              Arrays.asList(
                  categories.get(random.nextInt(categories.size())),
                  categories.get(random.nextInt(categories.size()))),
              locations.get(random.nextInt(locations.size())),
              riskLevels.get(random.nextInt(riskLevels.size())));

      profiles.add(profile);
      System.out.println(
          "Generated profile: "
              + profile.customerId()
              + " ($"
              + profile.averageTransactionAmount()
              + " avg, "
              + profile.riskLevel()
              + " risk)");
    }

    return profiles;
  }

  private static void sendCustomerProfiles(
      KafkaProducer<String, String> producer, List<CustomerProfile> profiles) throws Exception {
    for (CustomerProfile profile : profiles) {
      String json = objectMapper.writeValueAsString(profile);
      ProducerRecord<String, String> record =
          new ProducerRecord<>("customerProfiles", profile.customerId(), json);
      producer.send(record);
    }
    System.out.println("Sent " + profiles.size() + " customer profiles to Kafka");
  }

  // ================================
  // TRANSACTION GENERATION SCENARIOS
  // ================================

  /** Scenario 1: Normal legitimate transactions */
  private static void generateNormalTransactions(
      KafkaProducer<String, String> producer, List<CustomerProfile> profiles, int count)
      throws Exception {
    System.out.println("Generating " + count + " NORMAL transactions...");

    for (int i = 0; i < count; i++) {
      CustomerProfile profile = profiles.get(random.nextInt(profiles.size()));
      Transaction transaction = createNormalTransaction(profile);
      sendTransaction(producer, transaction);
      System.out.println(
          "Sent: "
              + transaction.transactionId()
              + " - $"
              + transaction.amount()
              + " at "
              + transaction.merchantCategory());
      Thread.sleep(1000); // 1-second delay
    }
  }

  /**
   * Scenario 2: High velocity attack (rapid-fire transactions). This triggers streaming velocity
   * intelligence
   */
  private static void generateHighVelocityAttack(
      KafkaProducer<String, String> producer, CustomerProfile profile, int count) throws Exception {
    System.out.println("Generating HIGH VELOCITY ATTACK (" + count + " rapid transactions)... for customer: " + profile.customerId());
    System.out.println("This will trigger streaming velocity intelligence!");

    for (int i = 0; i < count; i++) {
      Transaction transaction = createRapidFireTransaction(profile, i);
      sendTransaction(producer, transaction);
      System.out.println(
          "Rapid-fire #"
              + (i + 1)
              + ": "
              + transaction.transactionId()
              + " - $"
              + transaction.amount());
      Thread.sleep(200); // 200ms delay = 5 transactions per second
    }
  }

  /** Scenario 3: Unusual amount fraud. This triggers customer profile intelligence */
  private static void generateUnusualAmountFraud(
      KafkaProducer<String, String> producer, CustomerProfile profile, int count) throws Exception {
    System.out.println("Generating UNUSUAL AMOUNT fraud (" + count + " transactions)...");
    System.out.println("This will trigger customer profile intelligence!");

    for (int i = 0; i < count; i++) {
      Transaction transaction = createUnusualAmountTransaction(profile);
      sendTransaction(producer, transaction);
      System.out.println(
          "Unusual: "
              + transaction.transactionId()
              + " - $"
              + transaction.amount()
              + " (5x customer average!)");
      Thread.sleep(2000); // 2-second delay
    }
  }

  /** Scenario 4: Mixed scenario - combination of normal and fraudulent */
  private static void generateMixedScenario(
      KafkaProducer<String, String> producer, List<CustomerProfile> profiles) throws Exception {
    System.out.println("Generating MIXED scenario...");

    // Normal transactions
    for (int i = 0; i < 5; i++) {
      CustomerProfile profile = profiles.get(random.nextInt(profiles.size()));
      sendTransaction(producer, createNormalTransaction(profile));
      Thread.sleep(1000);
    }

    // High-velocity attack
    System.out.println("Injecting high velocity attack...");
    generateHighVelocityAttack(producer, profiles.getFirst(), 8);

    // More normal transactions
    for (int i = 0; i < 5; i++) {
      CustomerProfile profile = profiles.get(random.nextInt(profiles.size()));
      sendTransaction(producer, createNormalTransaction(profile));
      Thread.sleep(1000);
    }

    // Unusual amount
    System.out.println("\nInjecting unusual amount fraud...");
    generateUnusualAmountFraud(producer, profiles.get(1), 3);
  }

  /** Scenario 5: Continuous stream for real-time testing */
  private static void generateContinuousStream(
      KafkaProducer<String, String> producer, List<CustomerProfile> profiles) throws Exception {
    System.out.println("Generating CONTINUOUS stream... (Press Ctrl+C to stop)");

    int count = 0;
    while (true) {
      CustomerProfile profile = profiles.get(random.nextInt(profiles.size()));

      // 80% normal, 10% high velocity, 10% unusual amount
      Transaction transaction;
      int scenario = random.nextInt(100);

      if (scenario < 80) {
        transaction = createNormalTransaction(profile);
        System.out.println("Normal: " + transaction.transactionId());
      } else if (scenario < 90) {
        transaction = createRapidFireTransaction(profile, count);
        System.out.println("Velocity: " + transaction.transactionId());
      } else {
        transaction = createUnusualAmountTransaction(profile);
        System.out.println("Unusual: " + transaction.transactionId());
      }

      sendTransaction(producer, transaction);
      count++;

      Thread.sleep(500 + random.nextInt(1500)); // 0.5-2 second delay
    }
  }

  // ================================
  // TRANSACTION BUILDERS
  // ================================

  private static Transaction createNormalTransaction(CustomerProfile profile) {
    return new Transaction(
        "TXN-" + UUID.randomUUID().toString().substring(0, 8),
        profile.customerId(),
        profile.averageTransactionAmount().multiply(BigDecimal.valueOf(0.5 + random.nextDouble())),
        "USD",
        "MERCHANT-" + random.nextInt(1000),
        profile.transactionCategories().get(random.nextInt(profile.transactionCategories().size())),
        profile.primaryLocation(),
        LocalDateTime.now(),
        Map.of("channel", "ONLINE",
                "deviceId", "DEVICE-" + random.nextInt(100)));
  }

  private static Transaction createRapidFireTransaction(CustomerProfile profile, int sequence) {
    return new Transaction(
        "TXN-RAPID-" + UUID.randomUUID().toString().substring(0, 8),
        profile.customerId(),
        BigDecimal.valueOf(10 + random.nextInt(50)), // Small amounts for card testing
        "USD",
        "MERCHANT-SUSPICIOUS-" + random.nextInt(10),
        "ONLINE",
        "Unknown Location",
        LocalDateTime.now(),
        Map.of(
            "channel",
            "ONLINE",
            "rapidFire",
            true,
            "sequence",
            sequence,
            "deviceId",
            "BOT-DEVICE-" + random.nextInt(5)));
  }

  private static Transaction createUnusualAmountTransaction(CustomerProfile profile) {
    // 5x the average - unusual!
    BigDecimal unusualAmount = profile.averageTransactionAmount().multiply(BigDecimal.valueOf(5));

    return new Transaction(
        "TXN-UNUSUAL-" + UUID.randomUUID().toString().substring(0, 8),
        profile.customerId(),
        unusualAmount,
        "USD",
        "MERCHANT-LUXURY-" + random.nextInt(10),
        "LUXURY_GOODS",
        "Different Location",
        LocalDateTime.now(),
        Map.of(
            "channel", "ONLINE",
            "unusualAmount", true,
            "deviceId", "UNKNOWN-DEVICE"));
  }

  private static void sendTransaction(
      KafkaProducer<String, String> producer, Transaction transaction) throws Exception {
    String json = objectMapper.writeValueAsString(transaction);
    ProducerRecord<String, String> record =
        new ProducerRecord<>("transactions", transaction.customerId(), json);
    producer.send(record);
  }
}
