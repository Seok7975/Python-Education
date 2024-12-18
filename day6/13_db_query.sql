use sample_db;

#회원테이블(부모)
CREATE TABLE member
(
   id VARCHAR(30) NOT NULL,     
   pass VARCHAR(30) NOT NULL,    
   name VARCHAR(30) NOT NULL,  
   regidate datetime DEFAULT CURRENT_TIMESTAMP,  
   PRIMARY KEY (id)
);

#게시판테이블(자식)
CREATE TABLE board
(
   num INT NOT NULL AUTO_INCREMENT,            
   title VARCHAR(100) NOT NULL,   
   content TEXT NOT NULL, 
   id VARCHAR(30) NOT NULL,   
   postdate DATETIME DEFAULT current_timestamp, 
   visitcount MEDIUMINT NOT NULL DEFAULT 0,
   PRIMARY KEY (num)
);

ALTER TABLE board ADD constraint fk_board_member
   FOREIGN KEY (id) REFERENCES member (id);

